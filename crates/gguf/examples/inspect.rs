//! Header inspector: cargo run -p gguf --example inspect -- <path> [tensor-substr]
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: inspect <gguf-or-header> [tensor-substr]");
    let filt = args.next();
    let head = std::fs::read(&path).expect("read");
    let g = gguf::Gguf::parse(&head).expect("parse");
    println!("arch: {:?}", g.architecture());
    let mut kv: Vec<&String> = g.meta_keys();
    kv.sort();
    for k in kv {
        if !k.starts_with("tokenizer") {
            println!("kv {k} = {}", g.meta_display(k));
        }
    }
    for t in &g.tensors {
        let show = filt.as_deref().map_or(t.name.contains("blk.1.") || !t.name.contains("blk."),
                                          |f| t.name.contains(f));
        if show {
            println!("tensor {} {:?} {:?}", t.name, t.ty, t.dims);
        }
    }
    println!("({} tensors total)", g.tensors.len());
}
