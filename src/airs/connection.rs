use std::sync::Arc;

use super::neuron::Neuron;
use super::neuron::Type;
use super::neuron::Value;

pub struct Connection {
    neuron: Arc<Neuron>,
    inputs: Vec<Arc<Connection> >,
}

impl Connection {
    pub fn new(neuron: Arc<Neuron>, inputs: Vec<Arc<Connection> >) -> Self {
        Self {
            neuron,
            inputs,
        }
    }

    pub fn to_string(&self) -> String {
        if self.inputs.is_empty() {
            self.neuron.name()
        } else {
            let input_strs: Vec<String> = self.inputs
                .iter()
                .map(|conn| conn.to_string())
                .collect();
            format!("{}({})", self.neuron.name(), input_strs.join(", "))
        }
    }

    pub fn neuron(&self) -> Arc<Neuron> {
        Arc::clone(&self.neuron)
    }

    pub fn cost(&self) -> usize {
        let mut c = 0;

        for v in &self.inputs {
            c += v.cost() + 1;
        }

        c
    }

    pub fn depth(&self, d: usize) -> usize {
        self.inputs.iter().fold(d, |acc, v| {
            std::cmp::max(acc, v.depth(d + 1))
        })
    }

    pub fn output(&self) -> Option<Value> {
        let mut args = Vec::with_capacity(self.inputs.len());

        for v in &self.inputs {
            args.push(v.output()?);
        }

        self.neuron.apply(&args)
    }
/*
    pub fn apply_inputs(&mut self, inputs: Vec<Arc<Connection> >) {
        assert_eq!(inputs.len(), self.input_types().len());

        let mut index: usize = 0;

        for i in 0..self.inputs.len() {
            let input = &mut self.inputs[i];
            let mut conn = input.clone();
            let size = conn.input_types().len();

            if size > 0 {
                let sub_inputs = &inputs[index..index + size];

                conn.apply_inputs(sub_inputs.to_vec());
                *input = conn;
                index += size;
            } else {
                *input = inputs[index].clone();
                index += 1;
            }
        }
    }
*/
    pub fn input_types(&self) -> Vec<Type> {
        let mut types = Vec::with_capacity(self.inputs.len());

        for input in &self.inputs {
            let sub_types = input.input_types();

            if sub_types.is_empty() {
                types.push(input.neuron.output_type().clone());
            } else {
                types.extend(sub_types);
            }
        }

        types
    }
}
