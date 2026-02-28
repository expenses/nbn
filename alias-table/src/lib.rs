#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alias {
    pub prob: f32,
    pub pdf: f32,
    pub index: u32,
}

// Based on https://idarago.github.io/the-alias-method/
pub fn construct(values: &[f32]) -> Vec<Alias> {
    if values.is_empty() {
        return Vec::new();
    }

    let sum = values.iter().sum::<f32>();
    let mean = sum / values.len() as f32;

    let mut small_values = Vec::with_capacity(values.len() / 2);
    let mut large_values = Vec::with_capacity(values.len() / 2);
    let mut aliases = Vec::with_capacity(values.len());

    for (i, &value) in values.iter().enumerate() {
        let alias = Alias {
            prob: value / mean,
            pdf: value / sum,
            index: 0
        };
        aliases.push(alias);
        if alias.prob <= 1.0 {
            small_values.push(i);
        } else {
            large_values.push(i);
        }
    }

    while let (Some(large), Some(small)) = (large_values.pop(), small_values.pop()) {
        aliases[small].index = large as _;
        aliases[large].prob = aliases[large].prob + aliases[small].prob - 1.0;

        if aliases[large].prob < 1.0 {
            small_values.push(large);
        } else {
            large_values.push(large);
        }
    }

    for i in large_values {
        aliases[i].prob = 1.0;
    }
    for i in small_values {
        aliases[i].prob = 1.0;
    }

    aliases
}

#[test]
fn test_basic() {
    // Test with values from https://bfraboni.github.io/data/alias2022/alias-table.pdf
    assert_eq!(
        construct(&[7.0, 3.0, 4.0, 1.0, 6.0, 3.0]),
        [
            Alias {
                prob: 1.0,
                index: 0,
                pdf: 0.29166666
            },
            Alias {
                prob: 0.75,
                index: 0,
                pdf: 0.125
            },
            Alias {
                prob: 1.0,
                index: 0,
                pdf: 0.16666667
            },
            Alias {
                prob: 0.25,
                index: 4,
                pdf: 0.041666668
            },
            Alias {
                prob: 0.5,
                index: 0,
                pdf: 0.25
            },
            Alias {
                prob: 0.75,
                index: 4,
                pdf: 0.125
            }
        ]
    );
    assert_eq!(construct(&[]), []);
}
