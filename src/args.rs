use std::path::PathBuf;
use clap::{Args, Parser, Subcommand, ValueEnum};


#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// create an inference session
    /// [Note] Do not call start directly, call `run` command to inference and daemon precess will auto start.
    Start(StartArgs),
    /// run an inference session. If no daemon exists, create one.
    Run (RunArgs),
    /// stop daemon precess
    Stop
}

#[derive(Args)]
pub struct StartArgs {
    /// provider type [Cpu, DirectML]
    #[arg(short, long)]
    pub provider:Option<Provider>,
    /// preprocess type
    #[arg(short,long)]
    pub pp_type: Option<PreprocessType>,
    /// [Cpu only] intra node inference threads of decoder
    #[arg(long,default_value_t=8)]
    pub intra_threads:usize,
    /// [Cpu only] inter node inference threads of decoder
    #[arg(long,default_value_t=8)]
    pub inter_threads:usize,
}

#[derive(Args)]
pub struct RunArgs {
    /// input img path
    pub img_path: PathBuf,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum PreprocessType{
    Resize,
    Fill
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Provider {
    /// ort cpu, high speed with high cpu usage
    Cpu,
    /// windows.ai.ml, lower cpu with graph accelerator enabled, but may be slower.
    DirectML,
    /// Nvidia CUDA, not impl
    CUDA,
    /// Mac CoreML, not impl
    CoreML
}