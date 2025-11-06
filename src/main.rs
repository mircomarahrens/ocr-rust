use clap::Parser;
use rocr::nets::convolutional_layer::ConvolutionalLayer;
use rocr::nets::im2col;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;

use rocr::data::idx::{Idx1, Idx3};
use rocr::math::tensor::Tensor;

#[derive(Parser)]
struct Config {
    images_path: PathBuf,
    labels_path: PathBuf,
}

fn main() -> std::io::Result<()> {
    env_logger::init();

    // parse config
    let args = Config::parse();

    let images_path = args.images_path.to_str().unwrap();
    let labels_path = args.labels_path.to_str().unwrap();

    // read header of file for images
    let file = File::open(images_path).unwrap();
    let mut reader = BufReader::new(file);

    let mut current = reader.stream_position()?;
    println!("Current position: {}", current);

    // The header contains basic information about the images
    let header = Idx3::read_header(&mut reader)?;
    println!(
        "Magic number: {}, Num images: {}, rows: {}, cols: {}",
        header.magic_num, header.num_images, header.shape.0, header.shape.1
    );

    current = reader.stream_position()?;
    println!("Current position: {}", current);

    // TODO implement im2col following https://cs231n.github.io/convolutional-networks/

    //    // Read and process each image chunk in parallel
    //    let chunk_size = 1000; // Adjust the chunk size as needed
    //    let mut handles = vec![];
    //    loop {
    //        let mut chunk = vec![0u8; chunk_size * 28 * 28]; // Assuming MNIST images are 28x28 pixels
    //        match reader.read_exact(&mut chunk) {
    //            Ok(_) => {
    //                let handle = thread::spawn(move || {
    //                    process_chunk(chunk, chunk_size);
    //                });
    //
    //                handles.push(handle);
    //            }
    //            Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => break, // Break if end of file
    //            Err(e) => return Err(e), // Propagate other errors
    //        }
    //    }
    //
    //    // Wait for all threads to finish
    //    for handle in handles {
    //        handle.join().unwrap();
    //    }

    let data = Idx3::read_next_image(&header, &mut reader)?;
    print_image(data.clone());

    let data = Idx3::read_next_image(&header, &mut reader)?;
    print_image(data.clone());

    let data_f32 = data.iter().map(|x| *x as f32).collect();
    let tensor_a = Tensor::new(data_f32, vec![28, 28]);

    for i in 0..28 {
        for j in 0..28 {
            print!("{:4} ", tensor_a[&[i, j]]);
        }
        println!();
    }

    // Test
    let data_x = [1, 2, 3, 4, 5, 6];
    let data_f32 = data_x.iter().map(|x| *x as f32).collect();
    let mut matrix = Tensor::new(data_f32, vec![2, 3]);
    matrix.reshape(vec![2, 3]);

    println!();
    println!("=====================");
    println!();

    for i in 0..2 {
        for j in 0..3 {
            print!("{:4} ", matrix[&[i, j]]);
        }
        println!();
    }

    println!();
    println!("=====================");
    println!();

    // for i in matrix.strides() {
    //     print!("{:4} ", i);
    // }

    println!();
    println!("Old shape {:?}", matrix.shape());

    println!("Apply padding of 1 with 0.0");
    matrix.pad(1, 0.0);

    println!("New shape {:?}", matrix.shape());

    // for i in matrix.strides() {
    //     print!("{:4} ", i);
    // }

    println!();
    println!("=====================");
    println!();

    for i in 0..4 {
        for j in 0..5 {
            print!("{:4} ", matrix[&[i, j]]);
        }
        println!();
    }

    //    matrix.reshape(vec![5, 4]);
    //
    //    println!();
    //    println!("=====================");
    //    println!();
    //
    //    for i in 0..5 {
    //        for j in 0..4 {
    //            print!("{:4} ", matrix[&[i, j]]);
    //        }
    //        println!();
    //    }

    Ok(())
}

fn print_image(data: Vec<u8>) {
    print!("       ");
    for i in 0..28 {
        print!("{:4} ", i);
    }
    println!();
    print!("       ");
    for _i in 0..35 {
        print!("____");
    }
    println!();

    for i in 0..28 {
        print!("{:4} | ", i);
        for j in 0..28 {
            print!("{:4} ", data[i * 28 + j]);
        }
        println!();
    }
}

fn print_images(images: Vec<u8>, chunk_size: usize) {
    for c in 0..chunk_size {
        let offset = c * 28 * 28;
        print!("       ");
        for i in 0..28 {
            print!("{:4} ", i);
        }
        println!();
        print!("       ");
        for _i in 0..35 {
            print!("____");
        }
        println!();

        for i in 0..28 {
            print!("{:4} | ", i);
            for j in 0..28 {
                print!("{:4} ", images[i * 28 + j + offset]);
            }
            println!();
        }
    }
}
