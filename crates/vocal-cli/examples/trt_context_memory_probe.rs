//! Identify whether first JIT memory belongs to a context or its engine.
#[cfg(windows)]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vocal_tensorrt::{OwnedContext, Runtime};
    let plan = std::env::args_os().nth(1).ok_or("expected plan path")?;
    let runtime = Runtime::shared(plan)?;
    println!("runtime loaded: pid={}", std::process::id());
    for pass in 0..2 {
        let context = OwnedContext::create_with_cuda_graphs(runtime.clone(), true)?;
        println!("context {pass} created");
        std::thread::sleep(std::time::Duration::from_millis(1000));
        drop(context);
        println!("context {pass} dropped");
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
    Ok(())
}
