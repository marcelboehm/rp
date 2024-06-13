import os
import time
import gradio as gr
import rp.globals
import rp.metadata
import rp.utilities as util
import ui.globals as uii

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.livecam_tab import livecam_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab

rp.globals.keep_fps = None
rp.globals.keep_frames = None
rp.globals.skip_audio = None
rp.globals.use_batch = None


def prepare_environment():
    rp.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(rp.globals.output_path, exist_ok=True)
    if not rp.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'


def run():
    from rp.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)
    rp.globals.execution_providers = decode_execution_providers([rp.globals.CFG.provider])
    print(f'Using provider {rp.globals.execution_providers} - Device:{util.get_device()}')    
    
    run_server = True
    uii.ui_restart_server = False
    mycss = """
        span {color: var(--block-info-text-color)}
        #fixedheight {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
        .image-container.svelte-1l6wqyv {height: 100%}

    """

    while run_server:
        server_name = rp.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = rp.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = False if server_name == '0.0.0.0' else True
        with gr.Blocks(title=f'{rp.metadata.name} {rp.metadata.version}', theme=rp.globals.CFG.selected_theme, css=mycss) as ui:
            with gr.Row(variant='compact'):
                    gr.Markdown(f"### [{rp.metadata.name} {rp.metadata.version}](https://github.com/C0untFloyd/rp-unleashed)")
                    gr.HTML(util.create_version_html(), elem_id="versions")
            faceswap_tab()
            livecam_tab()
            facemgr_tab()
            extras_tab()
            settings_tab()

        uii.ui_restart_server = False
        try:
            ui.queue().launch(inbrowser=True, server_name=server_name, server_port=server_port, share=rp.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)

