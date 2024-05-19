# Vaughan

[![Crates.io](https://img.shields.io/crates/v/vaughan.svg)](https://crates.io/crates/vaughan)
[![Documentation](https://docs.rs/vaughan/badge.svg)](https://docs.rs/vaughan/)
[![Codecov](https://codecov.io/github/savente93/vaughan/coverage.svg?branch=master)](https://codecov.io/gh/savente93/vaughan)
[![Dependency status](https://deps.rs/repo/github/savente93/vaughan/status.svg)](https://deps.rs/repo/github/savente93/vaughan)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


Vaughan is a Rust library designed to provide fast, reliable, and ergonomic implementations of various scientific, statistical, and data science-related concepts.  Built on top of the powerful [Polars](https://github.com/pola-rs/polars) library, Vaughan leverages the speed and efficiency of Polars to deliver high-performance computations. This library is part of the ambition to bring the SciPy/Scikit-learn stack to Polars & Rust.  Vaughan is named in honor of [Dorothy Vaughan](https://en.wikipedia.org/wiki/Dorothy_Vaughan), one of the black women mathematician and human computer who worked for NASA during the Space Race and were invaluable to the moon landing. 


## Quick start 🚀

Add Vaughan to your `Cargo.toml` mannually:

```toml
[dependencies]
vaughan = "0.1.0"
```

or by using `cargo add`

```sh
cargo add vaughan
```

Currently, Vaughan is only available for use through Rust, but I am planning to add Python bindings soon.

Here's a simple example to get you started adapted from the test suite:

```rust
use vaughan::error_metrics::binary_classification::*;
use polars::prelude::*;

fn main() {

        let test = df!(
            "predictions" => [0,1,1,1,0,1],
            "truth" =>       [1,0,0,1,0,1]

        )?;
        println!("{}", accuracy(test.clone().lazy(), "predictions", "truth")?);
        println!("{}", f1(test.clone().lazy(), "predictions", "truth")?);
        println!("{}", recall(test.clone().lazy(), "predictions", "truth")?);
        println!("{}", precision(test.clone().lazy(), "predictions", "truth")?);
}
```

## Design

### Design goals

- Have great testing and documentation. Documentation is boring, but critical to a good user experience. It is also something where I am trying to improve, so this is an important focus of this project. (I'm working on it)
- Be fast. Part of why I think Polars and Rust are fertile soil for this is the speed they bring. My goal is to maintain the same speed in subsequent works wherever possible. 
- Don't be wrong. Scientific computations are complex and subtle, but very critical. If this library is going to be used by anyone for anything of note, it needs to be as correct as possible. Therefore, trying to make sure the code is as robustly tested as possible is important. 
- Foster a great community. While this is not something I can control directly, part of what attracted me to Rust was the welcoming community. I hope to replicate that here, by being as open to feedback as possible and fostering a welcoming community to all that want to be part of it. 

### Non goals

I believe that non goals are as important as goals for a good and focused design. Here are some non goals for this project: 

- Being "beginner-friendly". In my opinion, being beginner-friendly is important but should not be a design goal itself. This should be a byproduct of having good documentation and an intuitive API. 
- Deep learning & multidimensional data. The goal of this library is to leverage the power of Polars. The DL and multidimensional data model is different enough that I think Polars will not be a good fit, nor is it my expertise. Therefore, that functionality belongs in other libraries. 
- 100% compatibility/feature parity with SciPy or scikit-learn. While SciPy and Scikit-learn are great inspirations, they are also massive projects with their own goals. I will aim for as high of a feature coverage as I can, but I cannot promise to match them. Aside from that, the design of Polars is different to that of Pandas or NumPy and as a result the code built on top of it should be designed for it's way of doing things, not those that came before. 

## Roadmap

Vaughan very young, and currently not even ready to be called an alpha. It is little more than a prototype at this point. Therefore, expect breaking changes basically without warning and lots of missing features. However, I would very much like to expand it. [Feature Requests, Bug Reports](https://github.com/savente93/vaughan/issues), [Pull Requests](https://github.com/savente93/vaughan/pulls), [Questions](https://github.com/savente93/vaughan/discussions), and any other form of constructive feedback is very welcome! This is a spare time project for me, so I can't promise any timeline, but here is a rough outline of the things I am hoping to add: 

- [ ] Comprehensive documentation
- [ ] Python Bindings
- [ ] Benchmarks
- [ ] Statistical calculations
- [ ] data pre-processing procedures such as regularization 
- [ ] dimensionality reduction

## Acknowledgements

- [Polars](https://github.com/pola-rs/polars): for providing the foundation to build this code on top of.
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn) for serving as the inspiration and providing implementations and tests to benchmark and test against
- [SciPy](https://github.com/scipy/scipy) for serving as the inspiration and providing implementations and tests to benchmark and test against
- [Dorothy Vaughan](https://en.wikipedia.org/wiki/Dorothy_Vaughan) for being a pioneer in digital/computational mathematics and serving as an inspiration.
- [Jon Gjengset](https://github.com/jonhoo) for his excellent educational content in Rust and for providing the basis for the CI setup in his excellent repo (rust-ci-conf)[https://github.com/jonhoo/rust-ci-conf]

Thank you for using Vaughan! I hope it helps you with your scientific and data science endeavors.
