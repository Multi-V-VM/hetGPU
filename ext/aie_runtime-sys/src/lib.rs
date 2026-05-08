//! Raw XRT C-API bindings (bindgen-generated). No safe wrappers — consumers
//! build their own RAII types on top (see `zluda/src/impl/`).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
