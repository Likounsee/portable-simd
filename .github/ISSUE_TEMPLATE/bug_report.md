---
name: Bug Report
about: Create a bug report for Rust.
labels: C-bug
---
<!--
Thank you for filing a bug report! 🐛 Please provide a short summary of the bug,
along with any information you feel relevant to replicating the bug.
-->

I tried this code:

```rust
<code>
```

I expected to see this happen: *explanation*

Instead, this happened: *explanation*

### Meta

`rustc --version --verbose`:
```
<version>
```


`crate version in Cargo.toml`:
```toml
[dependencies]
core_simd = { git = "https://github.com/Likounsee/portable-simd" }
```
<!-- If this specifies the repo at HEAD, please include the latest commit. -->


<!--
If a backtrace is available, please include a backtrace in the code block by
setting `RUST_BACKTRACE=1` in your environment. e.g.
`RUST_BACKTRACE=1 cargo build`.
-->
<details><summary>Backtrace</summary>
<p>

```
<backtrace>
```

</p>
</details>
