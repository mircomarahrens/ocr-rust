use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

pub struct Idx1 {
    pub magic_num: i32,
    pub num_labels: i32,
}

impl Idx1 {
    pub fn new(magic_num: i32, num_labels: i32) -> Idx1 {
        Idx1 {
            magic_num,
            num_labels,
        }
    }

    pub fn read_header(reader: &mut BufReader<File>) -> std::io::Result<Idx1> {
        let magic_num = reader.read_i32::<BigEndian>()?;
        let num_labels = reader.read_i32::<BigEndian>()?;

        Ok(Idx1::new(magic_num, num_labels))
    }

    pub fn read_next_label(reader: &mut BufReader<File>) -> std::io::Result<u8> {
        let label = reader.read_u8()?;

        Ok(label)
    }
}

pub struct Idx3 {
    pub magic_num: i32,
    pub num_images: i32,
    pub shape: (i32, i32),
}

impl Idx3 {
    fn new(magic_num: i32, num_images: i32, shape: (i32, i32)) -> Idx3 {
        Idx3 {
            magic_num,
            num_images,
            shape,
        }
    }

    pub fn read_header(reader: &mut BufReader<File>) -> std::io::Result<Idx3> {
        match reader.stream_position()? {
            0 => (),
            _ => {
                reader.seek(std::io::SeekFrom::Start(0))?;
            }
        }

        let magic_num = reader.read_i32::<BigEndian>()?;
        let num_images = reader.read_i32::<BigEndian>()?;
        let num_rows = reader.read_i32::<BigEndian>()?;
        let num_cols = reader.read_i32::<BigEndian>()?;

        Ok(Idx3::new(magic_num, num_images, (num_rows, num_cols)))
    }

    pub fn read_next_image(&self, reader: &mut BufReader<File>) -> std::io::Result<Vec<u8>> {
        let mut img = vec![0u8; (self.shape.0 * self.shape.1) as usize];
        reader.read_exact(&mut img)?;

        Ok(img)
    }

    pub fn read_next_images(
        &self,
        reader: &mut BufReader<File>,
        num_images: usize,
    ) -> std::io::Result<Vec<u8>> {
        let mut images = vec![0u8; num_images * (self.shape.0 * self.shape.1) as usize];
        reader.read_exact(&mut images)?;

        Ok(images)
    }
}
