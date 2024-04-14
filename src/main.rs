use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::io::BufWriter;
use algorithm::ReverseSearchState;
use clap::Parser;
use std::vec::Vec;
use std::string::String;
use algorithm::FullPolytope;
use anyhow::Result;
use simplelog::*;
use crate::algorithm::reverse_search;
use log::{debug, info};

pub mod algorithm;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    polytope_file: String,

    #[arg(long)]
    polytope_out: String,

    #[arg(long)]
    reserve_search_out: String,
}

fn read_polytope(poly_str: String) -> serde_json::Result<Vec<FullPolytope>>{
    let deserialised: Vec<FullPolytope> = serde_json::from_str(&poly_str)?;
    eprintln!("Loaded {} polytopes", deserialised.len());
    Ok(deserialised)
}

fn write_polytope(poly_str: &Vec<FullPolytope>, out_filename: &String) -> Result<()>{
    eprintln!("Saving {} polytopes to {}", poly_str.len(), out_filename);
    let out_string = serde_json::to_string_pretty(poly_str)?;
    let poly_out_file = File::create(out_filename)?;
    let mut writer = BufWriter::new(poly_out_file);    
    writer.write(out_string.as_bytes())?;
    return Ok(());
}

fn write_search_results(states: &Vec<(ReverseSearchState, f64)>, out_filename: &String) -> Result<()>{
    info!("Saving {} search results to {}", states.len(), out_filename);
    let out_string = serde_json::to_string_pretty(states)?;
    let states_out_file = File::create(out_filename)?;
    let mut writer = BufWriter::new(states_out_file);    
    writer.write(out_string.as_bytes())?;
    return Ok(());
}



fn main() -> Result<()>{
    TermLogger::init(LevelFilter::Debug, Config::default(), TerminalMode::Stderr, ColorChoice::Auto)?;

    let args = Args::parse();

    eprintln!("Loading {}!", args.polytope_file);

    let json_file = File::open(args.polytope_file)?;
    let mut buf_reader = BufReader::new(json_file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    let mut poly = read_polytope(contents)?;
    let states = reverse_search(&mut poly)?;
    write_polytope(&poly, &args.polytope_out)?;
    write_search_results(&states, &args.reserve_search_out)?;
    return Ok(());
}

