#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
import torch
import onnxruntime
import pathlib

from time import time

import rp.globals
import rp.metadata
import rp.utilities as util
import rp.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from rp.face_util import extract_face_images
from rp.ProcessEntry import ProcessEntry
from rp.ProcessMgr import ProcessMgr
from rp.ProcessOptions import ProcessOptions
from rp.capturer import get_video_frame_total


clip_text = None

call_display_ui = None

process_mgr = None


if 'ROCMExecutionProvider' in rp.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    rp.globals.headless = False
    # Always enable all processors when using GUI
    if len(sys.argv) > 1:
        print('No CLI args supported - use Settings Tab instead')
    rp.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in rp.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in rp.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if rp.globals.max_memory:
        memory = rp.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = rp.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources() -> None:
    import gc
    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None

    gc.collect()
    # if 'CUDAExecutionProvider' in rp.globals.execution_providers and torch.cuda.is_available():
    #     with torch.cuda.device('cuda'):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx'])
    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])
    download_directory_path = util.resolve_relative_path('../models/Frame')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if rp.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(rp.globals.source_path,  (False, 0))
        # rp.globals.INPUT_FACES.append(faces[rp.globals.source_face_index])
        # faces = extract_face_images(rp.globals.target_path,  (False, util.has_image_extension(rp.globals.target_path)))
        # rp.globals.TARGET_FACES.append(faces[rp.globals.target_face_index])
        # if 'face_enhancer' in rp.globals.frame_processors:
        #     rp.globals.selected_enhancer = 'GFPGAN'
       
    batch_process_regular(None, False, None)


def get_processing_plugins(masking_engine):
    processors = {  "faceswap": {}}
    if masking_engine is not None:
        processors.update({masking_engine: {}})
    
    if rp.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif rp.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif rp.globals.selected_enhancer == 'DMDNet':
        processors.update({"dmdnet": {}})
    elif rp.globals.selected_enhancer == 'GPEN':
        processors.update({"gpen": {}})
    elif rp.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})
    return processors


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    
#    if len(rp.globals.INPUT_FACESETS) <= selected_index:
#        selected_index = 0
    process_mgr.initialize(rp.globals.INPUT_FACESETS, rp.globals.TARGET_FACES, options)
    newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def batch_process_regular(files:list[ProcessEntry], masking_engine:str, new_clip_text:str, use_new_method, imagemask, num_swap_steps, progress, selected_index = 0) -> None:
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    mask = imagemask["layers"][0] if imagemask is not None else None
    if len(rp.globals.INPUT_FACESETS) <= selected_index:
        selected_index = 0
    options = ProcessOptions(get_processing_plugins(masking_engine), rp.globals.distance_threshold, rp.globals.blend_ratio, rp.globals.face_swap_mode, selected_index, new_clip_text, mask, num_swap_steps, False)
    process_mgr.initialize(rp.globals.INPUT_FACESETS, rp.globals.TARGET_FACES, options)
    batch_process(files, use_new_method)
    return

def batch_process_with_options(files:list[ProcessEntry], options, progress):
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    process_mgr.initialize(rp.globals.INPUT_FACESETS, rp.globals.TARGET_FACES, options)
    rp.globals.keep_frames = False
    rp.globals.wait_after_extraction = False
    rp.globals.skip_audio = False
    batch_process(files, True)



def batch_process(files:list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    rp.globals.processing = True

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        rp.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')


    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, rp.globals.output_path, f'.{rp.globals.CFG.output_image_format}')
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            destination = util.get_destfilename_from_path(fullname, rp.globals.output_path, f'__temp.{rp.globals.CFG.output_video_format}')
            f.finalname = destination
            videofiles.append(f)



    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.run_batch(origimages, fakeimages, rp.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not rp.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')
            start_processing = time()
            if rp.globals.keep_frames or not use_new_method:
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not rp.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, rp.globals.execution_threads)
                if not rp.globals.processing:
                    end_processing('Processing stopped!')
                    return
                if rp.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                ffmpeg.create_video(v.filename, v.finalname, fps)
                if not rp.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = rp.globals.skip_audio
                process_mgr.run_batch_inmem(v.filename, v.finalname, v.startframe, v.endframe, fps,rp.globals.execution_threads, skip_audio)
                
            if not rp.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']):
                    gifname = util.get_destfilename_from_path(v.filename, rp.globals.output_path, '.gif')
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    update_status('Creating final GIF')
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = rp.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)
                update_status(f'\nProcessing {os.path.basename(destination)} took {time() - start_processing} secs')

            else:
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
    end_processing('Finished')


def end_processing(msg:str):
    update_status(msg)
    rp.globals.target_folder_path = None
    release_resources()


def destroy() -> None:
    if rp.globals.target_path:
        util.clean_temp(rp.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    rp.globals.CFG = Settings('config.yaml')
    rp.globals.execution_threads = rp.globals.CFG.max_threads
    rp.globals.video_encoder = rp.globals.CFG.output_video_codec
    rp.globals.video_quality = rp.globals.CFG.video_quality
    rp.globals.max_memory = rp.globals.CFG.memory_limit if rp.globals.CFG.memory_limit > 0 else None
    main.run()
