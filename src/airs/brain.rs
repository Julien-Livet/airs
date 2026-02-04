use std::sync::Arc;

use super::neuron::Neuron;

pub struct Brain {
    neurons: Vec<Arc<Neuron> >,
}

impl Brain {
    pub fn new(neurons: Vec<Arc<Neuron> >) -> Self {
        Self
        {
            neurons,
        }
    }
}
