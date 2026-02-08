use ndarray::Array2;
use ndarray::prelude::*;
use reqwest;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RawTask {
    pub train: Vec<RawPair>,
    pub test: Vec<RawPair>,
}

#[derive(Debug, Deserialize)]
struct RawPair {
    pub input: Vec<Vec<i8> >,
    pub output: Option<Vec<Vec<i8> > >,
}


#[derive(Clone, Debug)]
pub struct Task {
    pub train: Vec<(Array2<i8>, Array2<i8>)>,
    pub test: Vec<(Array2<i8>, Option<Array2<i8> >)>,
}


#[derive(Clone, Debug)]
pub struct Pairs {
    pub inputs: Vec<Array2<i8> >,
    pub outputs: Vec<Array2<i8> >,
}

pub async fn load_task(folder: &str, task: &str) -> Result<Task, Box<dyn std::error::Error> > {
    let url = format!(
        "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/refs/heads/main/data/{}/{}.json",
        folder, task
    );
    let resp = reqwest::get(&url).await?;
    let raw_task: RawTask = resp.json().await?;
    let mut task = Task {
        train: Vec::new(),
        test: Vec::new(),
    };

    for ex in &raw_task.train {
        let arr_input = Array2::from_shape_vec(
            (ex.input.len(), ex.input[0].len()),
            ex.input.iter().flatten().cloned().collect(),
        ).unwrap();

        if let Some(ref out) = ex.output {
            let arr_output = Array2::from_shape_vec(
                (out.len(), out[0].len()),
                out.iter().flatten().cloned().collect(),
            ).unwrap();

            task.train.push((arr_input, arr_output));
        }
        else {
            task.train.push((arr_input, Array2::default((0, 0))));
        }
    }

    for ex in &raw_task.test {
        let arr_input = Array2::from_shape_vec(
            (ex.input.len(), ex.input[0].len()),
            ex.input.iter().flatten().cloned().collect(),
        ).unwrap();

        if let Some(ref out) = ex.output {
            let arr_output = Array2::from_shape_vec(
                (out.len(), out[0].len()),
                out.iter().flatten().cloned().collect(),
            ).unwrap();

            task.test.push((arr_input, Some(arr_output)));
        }
        else {
            task.test.push((arr_input, None));
        }
    }

    Ok(task)
}

pub fn input_output_pairs(pairs: &[(Array2<i8>, Array2<i8>)]) -> Pairs {
    let mut io_pairs = Pairs{inputs: vec![], outputs: vec![]};

    for p in pairs {
        let (input, output) = p;

        io_pairs.inputs.push(input.clone());
        io_pairs.outputs.push(output.clone());
    }

    io_pairs
}

pub fn input_option_output_pairs(pairs: &[(Array2<i8>, Option<Array2<i8> >)]) -> Pairs {
    let mut io_pairs = Pairs{inputs: vec![], outputs: vec![]};

    for p in pairs {
        let (input, output) = p;

        io_pairs.inputs.push(input.clone());
        io_pairs.outputs.push(output.clone().unwrap_or(array![[]]));
    }

    io_pairs
}
