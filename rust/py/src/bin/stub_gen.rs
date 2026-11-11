//! Writes `src/kibends/_core.pyi` from the extension module's stub information.

fn main() -> pyo3_stub_gen::Result<()> {
    _core::stub_info()?.generate()
}
