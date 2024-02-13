use resin::point::Point;

use clap::Parser;
use polars::prelude::*;
use indicatif::ProgressBar;
use itertools::Itertools;
use rand::random;
//use rayon::prelude::*;
//use rayon::ThreadPoolBuilder;

use std::fs::File;
use std::path::{Path, PathBuf};

/// CLI
#[derive(clap::Parser, Debug, Clone)]
#[clap( name  = "analytical"
      , about = "Create data based on an analytical")]
pub struct Cli {
    /// Out put file
    #[clap(short = 'o', long, default_value = "data/test/out.parquet")]
    pub outfile: PathBuf,

    /// Number of events per file
    #[clap(short = 'f', long, default_value = "1000")]
    pub evt_per_file: u64,

    /// Number of threads to use
    #[clap(short = 'j', long, default_value = "4")]
    pub threads: u8,

    /// Base of the total number of events in n=base**exponent
    #[clap(short = 'b', long, default_value = "10")]
    pub base: u64,

    /// Exponent of the total number of events in n=base**exponent
    #[clap(short = 'e', long)]
    pub exponent: u32,

    /// Exponent of the total number of events in n=base**exponent
    #[clap(short = 's', long, default_value = "0")]
    pub file_offset: u64,

    /// Active volume radius in mm
    #[clap(short='r', long, default_value = "200")]
    pub r: f64,

    /// Flag: Print SiPM positions.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    pub print_sipm_positions: bool,
}

/// Generate a new filename by appending _{index} before the file
/// extension
fn new_filename(filename: &Path, index: u64) -> PathBuf {
    let filename    = filename.to_str().unwrap();
    let split_index = filename.rfind(".").expect("Pattern not found");
    let (basename, extension) = filename.split_at(split_index);
    let new_file = PathBuf::from(format!("{basename}_{index}{extension}"));
    new_file
}

/// Generates a position in the range [-half_range, half_range) in
/// each dimension
fn generate_random_position(full_width: f64) -> Point {
    let x = full_width * (random::<f64>() - 0.5);
    let y = full_width * (random::<f64>() - 0.5);
    Point{x, y}
}

/// Evaluates the PSF at p1 with origin at p0
fn psf(p1: Point, p0: Point) -> f64 {
    let dp = p1 - p0;
    let dz2 = 5_f64.powi(2);

    dz2.powf(1.5) / (dp.r2() + dz2).powf(1.5)
}

/// Applies the PSF with origin at p0 to each point in sipm_pos.
/// Points falling outside of a circle with r2=rmax2 after applying
/// the translation trans are set to 0.
fn apply_psf(p0: Point, sipm_pos: &Vec<Point>, trans: Point, rmax2: f64) -> Vec<f64> {
    let mut response: Vec<f64> =
        sipm_pos.iter()
                .map(|&ps| {
                    if   (ps + trans).r2() < rmax2 { psf(ps, p0) }
                    else                           {      0_f64  }
                })
                .collect();

    response.insert(0, p0.y + trans.y);
    response.insert(0, p0.x + trans.x);
    response.insert(0, p0.y);
    response.insert(0, p0.x);
    response
}

/// Transform sequences of rows into a DataFrame
fn create_df(data: Vec<Vec<f64>>, names: &Vec<String>) -> DataFrame {
    let get_field = |i| {
        data.iter()
            .map(|v| v[i])
            .collect::<Vec<f64>>()
    };
    let columns: Vec<Series> =
    names.iter()
         .enumerate()
         .map(|(i, name)| (name, get_field(i)))
         .map(|(name, v)| Series::new(name, &v))
         .collect();

    DataFrame::new(columns).unwrap()
}

/// Generate the array of sipms based on the number of sipms per side
/// n and the pitch p
fn sipm_positions(n: usize, p: f64) -> Vec<Point> {
    let nf = n as f64;
    let x0 = -( p/2. + (nf-2.)/2.*p);
    (0..n).cartesian_product(0..n)
           .map(|(i, j)| {( i as f64, j as f64)})
           .map(|(i, j)| {( x0 + i*p, x0 + j*p)})
           .map(|(x, y)| {Point{x, y}})
           .collect()
}

/// Pick a random reference point in between 4 sipms
/// x,y = k*pitch, where k is an integer
fn pick_ref(p: f64, rmax: f64) -> Point {
    let range = 2. * (rmax + p);
    loop {
        let Point{mut x, mut y} = generate_random_position(range);
        x = x.div_euclid(p) * p;
        y = y.div_euclid(p) * p;
        let p = Point{x, y};
        if p.r() < rmax {return p;}
    }
}

fn get_column_names(nsipms: usize) -> Vec<String> {
    let mut column_names = Vec::with_capacity(4 + nsipms);
    column_names.push("x".to_string());
    column_names.push("y".to_string());
    column_names.push("xabs".to_string());
    column_names.push("yabs".to_string());
    (0..nsipms).into_iter()
               .map(|i| format!("sipm_{i}").to_string())
               .for_each(|name| column_names.push(name));

    column_names
}


fn main() -> Result<(), String> {
    let args = Cli::parse();
    println!("Arguments passed:{:?}", args);

    let filename = args.outfile;
    std::fs::create_dir_all(&filename.parent().expect("Could not access parent directory"))
        .expect("Cannot write to destination");

    let nsipms   = 16; // per side, this will produce a (nsipms x nsipms) response matrix
    let pitch    = 10.;
    let sipm_pos = sipm_positions(nsipms, pitch);
    if args.print_sipm_positions {
        for pos in &sipm_pos {
            println!("{} {}", pos.x, pos.y)
        }
    }

    let ntot   = args.base.pow(args.exponent);
    let nfile  = args.evt_per_file.min(ntot);
    let nbatch = (ntot as f64 / nfile as f64).round() as u64;
    assert_eq!(nbatch * nfile, ntot, "Invalid ratio of evt_per_file and ntot");

    let r  = args.r;
    let r2 = r * r;
    let pb = ProgressBar::new(nbatch); pb.set_position(0);
    // ThreadPoolBuilder::new()
    //     .num_threads(args.threads as usize)
    //     .build_global()
    //     .unwrap();

    let column_names = get_column_names(nsipms*nsipms);

    (0..nbatch).into_iter()
        .map(|i| new_filename(&filename, i + args.file_offset))
        .map(|filename| {
            let data: Vec<Vec<f64>> =
            (0..nfile).into_iter()
                      .map     (|_          | { (generate_random_position(pitch), pick_ref(pitch, r)) })
                      .map     (|(pos, refp)| { apply_psf(pos, &sipm_pos, refp, r2)                   })
                      .collect();

            let mut df   = create_df(data, &column_names);
            let mut file = File::create(filename).unwrap();
            ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        })
        .for_each(|_| { pb.inc(1);});

    Ok(())
}
