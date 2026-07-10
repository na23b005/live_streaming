use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandChild;
use std::sync::Mutex;

struct SidecarState(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .manage(SidecarState(Mutex::new(None)))
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      } else {
        // Automatically spawn python backend sidecar in production release builds
        let shell = app.shell();
        let sidecar = shell.sidecar("nexus-ai-backend").map_err(|e| {
          let err_msg = format!("failed to find sidecar 'nexus-ai-backend': {}", e);
          Box::<dyn std::error::Error>::from(err_msg)
        })?;
        let (_rx, child) = sidecar.spawn().map_err(|e| {
          let err_msg = format!("failed to spawn sidecar: {}", e);
          Box::<dyn std::error::Error>::from(err_msg)
        })?;

        // Store child process handle in Tauri state
        let state = app.state::<SidecarState>();
        *state.0.lock().unwrap() = Some(child);
      }
      Ok(())
    })
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|app_handle, event| {
      if let tauri::RunEvent::Exit = event {
        let state = app_handle.state::<SidecarState>();
        let mut guard = state.0.lock().unwrap();
        if let Some(child) = guard.take() {
          let _ = child.kill();
        }
      }
    });
}
