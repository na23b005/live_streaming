use tauri_plugin_shell::ShellExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
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
        let sidecar = shell.sidecar("local-transcribe-backend").map_err(|e| {
          let err_msg = format!("failed to find sidecar 'local-transcribe-backend': {}", e);
          Box::<dyn std::error::Error>::from(err_msg)
        })?;
        let (_rx, _tx) = sidecar.spawn().map_err(|e| {
          let err_msg = format!("failed to spawn sidecar: {}", e);
          Box::<dyn std::error::Error>::from(err_msg)
        })?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
