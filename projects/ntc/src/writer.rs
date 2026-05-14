pub struct AntcTexture<'a> {
    pub size: u32,
    pub block_offsets: &'a [u32],
    pub bytes: &'a [u8],
}

pub struct AntcWriter<'a> {
    pub weights: &'a [u8],
    pub textures: &'a [AntcTexture<'a>; 4],
}

impl<'a> AntcWriter<'a> {
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(b"ANTC")?;
        writer.write_all(&0_u32.to_le_bytes())?;

        let mut offset = 4
            + 4
            + 4
            + self
                .textures
                .iter()
                .map(|tex| 4 + 4 + tex.block_offsets.len() as u32 * 4)
                .sum::<u32>();

        writer.write_all(&offset.to_le_bytes())?;

        offset += self.weights.len() as u32;

        for texture in self.textures {
            writer.write_all(&texture.size.to_le_bytes())?;
            writer.write_all(&(texture.block_offsets.len() as u32).to_le_bytes())?;
        }

        for texture in self.textures {
            for block_offset in texture.block_offsets {
                writer.write_all(&(offset + block_offset).to_le_bytes())?;
            }
            offset += texture.bytes.len() as u32;
        }

        writer.write_all(&self.weights)?;
        for texture in self.textures {
            writer.write_all(&texture.bytes)?;
        }

        Ok(())
    }
}

#[test]
fn text_writer() {
    let writer = AntcWriter {
        weights: &[1, 3, 5, 7, 9],
        textures: [
            AntcTexture {
                size: 2,
                block_offsets: &[0, 4],
                bytes: &[0, 1, 2, 3, 5],
            },
            AntcTexture {
                size: 2,
                block_offsets: &[0, 8],
                bytes: nbn::cast_slice(&[6, 7, 8, 9, 77_u16]),
            },
            AntcTexture {
                size: 1,
                block_offsets: &[0],
                bytes: &[33],
            },
            AntcTexture {
                size: 1,
                block_offsets: &[0],
                bytes: &[44],
            },
        ],
    };

    let mut output = Vec::new();

    writer.write(&mut output).unwrap();

    assert_eq!(&output[..4], b"ANTC");

    let values: &[u32] = nbn::cast_slice(&output[4..]);

    let version = values[0];
    assert_eq!(version, 0);

    let weight_offset = values[1] as usize;

    assert_eq!(&output[weight_offset..weight_offset + 5], &[1, 3, 5, 7, 9]);

    let texture_info: &[[u32; 2]] = nbn::cast_slice(&values[2..2 + 4 * 2]);

    assert_eq!(texture_info, &[[2; 2], [2; 2], [1; 2], [1; 2]]);

    let mip_offsets_for_texture_2 = &values[2 + 4 * 2 + 2..2 + 4 * 2 + 2 + 2];

    assert_eq!(
        &nbn::cast_slice::<_, u16>(
            &output
                [mip_offsets_for_texture_2[0] as usize..mip_offsets_for_texture_2[0] as usize + 8]
        ),
        &[6, 7, 8, 9]
    );
    assert_eq!(
        &nbn::cast_slice::<_, u16>(
            &output
                [mip_offsets_for_texture_2[1] as usize..mip_offsets_for_texture_2[1] as usize + 2]
        ),
        &[77]
    );
}
