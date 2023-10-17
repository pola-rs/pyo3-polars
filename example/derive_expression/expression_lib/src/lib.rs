mod distances;
mod expressions;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;
