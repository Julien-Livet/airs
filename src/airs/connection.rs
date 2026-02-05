use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::os::unix::raw::ino_t;
use std::sync::{Arc, RwLock};

use super::neuron::Neuron;
use super::neuron::ValueType;
use super::neuron::NeuronValue;
use super::utility::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConnectionValue {
    Value(NeuronValue),
    Connection(Arc<Connection>),
}

#[derive(Debug)]
pub struct Connection {
    neuron: Arc<Neuron>,
    inputs: RwLock<Vec<ConnectionValue> >,
}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        if (self.neuron().deref() as *const Neuron) != (other.neuron().deref() as *const Neuron) {
            return false;
        }

        let inputs = self.inputs.read().expect("Lock poisoned");
        let other_inputs = other.inputs.read().expect("Lock poisoned");

        if inputs.len() != other_inputs.len() {
            return false;
        }

        let mut equal = true;

        for i in 0..inputs.len() {
            match (inputs[i].clone(), other_inputs[i].clone()) {
                (ConnectionValue::Connection(a), ConnectionValue::Connection(b)) => equal &= a == b,
                (ConnectionValue::Value(a), ConnectionValue::Value(b)) => equal &= a == b,
                (_, _) => equal = false,
            }

            if !equal {
                break;
            }
        }
        
        return equal;
    }
}

impl Eq for Connection {}

impl Hash for Connection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.neuron().deref() as *const Neuron).hash(state);

        let inputs = self.inputs.read().expect("Lock poisoned");

        for input in inputs.iter() {
            match input {
                ConnectionValue::Connection(c) => c.hash(state),
                ConnectionValue::Value(v) => v.hash(state),
            }
        }
    }
}

impl Connection {
    pub fn new(neuron: Arc<Neuron>, inputs: &[ConnectionValue]) -> Self {
        Self {
            neuron: neuron,
            inputs: RwLock::new(inputs.to_vec()),
        }
    }

    pub fn to_string(&self) -> String {
        let args: Vec<String> = self.inputs.read().expect("Lock poisoned").iter().map(|v| match v {
            ConnectionValue::Value(value) => match value {
                NeuronValue::Bool(b) => b.to_string(),
                NeuronValue::Char(s) => s.clone(),
                NeuronValue::Int32(i) => i.to_string(),
                NeuronValue::Int64(i) => i.to_string(),
                NeuronValue::Float(f) => f.to_string(),
                NeuronValue::Double(d) => d.to_string(),
                NeuronValue::Grid(g) => matrix_to_string(g),
                NeuronValue::String(s) => s.clone(),
                NeuronValue::ValueType(t) => format!("{:?}", t),
            }

            ConnectionValue::Connection(c) => c.to_string(),
        }).collect();

        let name = self.neuron.name().to_string();
        if args.is_empty() {
            name
        } else {
            format!("{}({})", name, args.join(", "))
        }
    }

    pub fn neuron(&self) -> Arc<Neuron> {
        Arc::clone(&self.neuron)
    }

    pub fn inputs(&self) -> Vec<ConnectionValue> {
        self.inputs.read().expect("Lock poisoned").to_vec()
    }

    pub fn cost(&self) -> usize {
        let mut c = 0;
        let inputs = self.inputs.read().expect("Lock poisoned");

        for v in inputs.iter() {

            c += 1;

            if let ConnectionValue::Connection(inner) = v {
                c += inner.cost();
            }
        }
        c
    }

    pub fn depth(&self, d: usize) -> usize {
        let inputs = self.inputs.read().expect("Lock poisoned");

        inputs.iter().fold(d, |acc, v| {
            if let ConnectionValue::Connection(inner) = v {
                std::cmp::max(acc, inner.depth(d + 1))
            } else {
                acc
            }
        })
    }

    pub fn output(&self) -> Option<NeuronValue> {
        let mut args = Vec::with_capacity(self.inputs.read().expect("Lock poisoned").len());
        let inputs = self.inputs.read().expect("Lock poisoned");

        for v in inputs.iter() {
            match v {
                ConnectionValue::Connection(inner) => {
                    args.push(inner.output()?);
                }
                ConnectionValue::Value(value) => args.push(value.clone()),
            }
        }

        self.neuron.apply(&args)
    }

    pub fn apply_inputs(&self, inputs: &[ConnectionValue]) {
        assert_eq!(inputs.len(), self.input_types().len());

        let mut inner_inputs = self.inputs.write().expect("Lock poisoned");
        let mut index: usize = 0;

        for i in 0..inner_inputs.len() {
            let input: &mut ConnectionValue = &mut inner_inputs[i];

            match input {
                ConnectionValue::Connection(conn_arc) => {
                    let conn = Arc::clone(conn_arc);
                    let size = conn.input_types().len();

                    if size > 0 {
                        let sub_inputs = &inputs[index..index + size];

                        conn.apply_inputs(sub_inputs);
                        index += size;
                    } else {
                        *input = inputs[index].clone();
                        index += 1;
                    }
                }
                _ => {
                    if index < inputs.len() {
                        match &inputs[index] {
                            ConnectionValue::Connection(_) => {
                            }
                            ConnectionValue::Value(value) => {
                                match value {
                                    NeuronValue::ValueType(t) => {
                                        assert_eq!(
                                            self.neuron.input_types()[i],
                                            *t
                                        );
                                    }
                                    other => {
                                        assert_eq!(
                                            self.neuron.input_types()[i],
                                            other.value_type()
                                        );
                                    }
                                }
                            }
                        }

                        *input = inputs[index].clone();
                        index += 1;
                    }
                }
            }
        }
    }

    pub fn input_types(&self) -> Vec<ValueType> {
        let inputs = self.inputs.read().expect("Lock poisoned");
        let mut types = Vec::with_capacity(inputs.len());

        for input in inputs.iter() {
            match input {
                ConnectionValue::Connection(conn) => {
                    let sub_types = conn.input_types();

                    if sub_types.is_empty() {
                        types.push(conn.neuron.output_type().clone());
                    } else {
                        types.extend(sub_types);
                    }
                }

                ConnectionValue::Value(value) => {
                    match value {
                        NeuronValue::ValueType(t) => {
                            types.push(t.clone());
                        }

                        other => {
                            types.push(other.value_type());
                        }
                    }
                }
            }
        }

        types
    }
}
