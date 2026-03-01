#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alias {
    pub threshold: f32,
    pub inv_pdf: f32,
    pub fallback: u32,
}

// Based on https://idarago.github.io/the-alias-method/
pub fn construct<I: IntoIterator<Item = (f32, f32)>>(values: I) -> Vec<Alias> {
    let mut sum = 0.0;

    let mut aliases: Vec<_> = values
        .into_iter()
        .map(|(weight, sample_weight)| {
            sum += weight;

            Alias {
                threshold: weight,
                // Some values should give lower values even after adjusting for the weight.
                // e.g. values at high latitudes in environment maps need to be smaller as the pixels are thinner.
                inv_pdf: sample_weight,
                fallback: 0,
            }
        })
        .collect();

    let mean = sum / aliases.len() as f32;

    let mut small_values = Vec::with_capacity(aliases.len() / 2);
    let mut large_values = Vec::with_capacity(aliases.len() / 2);

    for (i, alias) in aliases.iter_mut().enumerate() {
        let weight = alias.threshold;
        alias.threshold /= mean;
        alias.inv_pdf *= if weight == 0.0 { 0.0 } else { mean / weight };
        if alias.threshold <= 1.0 {
            small_values.push(i);
        } else {
            large_values.push(i);
        }
    }

    while let (Some(large), Some(small)) = (large_values.pop(), small_values.pop()) {
        aliases[small].fallback = large as _;
        aliases[large].threshold = aliases[large].threshold + aliases[small].threshold - 1.0;

        if aliases[large].threshold < 1.0 {
            small_values.push(large);
        } else {
            large_values.push(large);
        }
    }

    for i in large_values {
        aliases[i].threshold = 1.0;
    }
    for i in small_values {
        aliases[i].threshold = 1.0;
    }

    aliases
}

#[test]
fn test_basic() {
    // Test with values from https://bfraboni.github.io/data/alias2022/alias-table.pdf
    assert_eq!(
        construct(
            [7.0_f32, 3.0, 4.0, 1.0, 6.0, 3.0]
                .iter()
                .copied()
                .map(|x| (x, 1.0))
        ),
        [
            Alias {
                threshold: 1.0,
                inv_pdf: 0.5714286,
                fallback: 0
            },
            Alias {
                threshold: 0.75,
                inv_pdf: 1.3333334,
                fallback: 0
            },
            Alias {
                threshold: 1.0,
                inv_pdf: 1.0,
                fallback: 0
            },
            Alias {
                threshold: 0.25,
                inv_pdf: 4.0,
                fallback: 4
            },
            Alias {
                threshold: 0.5,
                inv_pdf: 0.6666667,
                fallback: 0
            },
            Alias {
                threshold: 0.75,
                inv_pdf: 1.3333334,
                fallback: 4
            }
        ]
    );
    assert_eq!(construct(std::iter::empty()), []);
}
