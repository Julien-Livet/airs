use std::sync::Arc;
use std::cell::RefCell;

use super::neuron::Neuron;
use super::neuron::Type;
use super::neuron::NeuronValue;

#[derive(Clone)]
pub enum ConnectionValue {
    Value(NeuronValue),
    Connection(Arc<Connection>),
}

#[derive(Clone)]
pub struct Connection {
    neuron: Arc<Neuron>,
    inputs: RefCell<Vec<ConnectionValue> >,
}

impl Connection {
    pub fn new(neuron: Arc<Neuron>, inputs: &[ConnectionValue]) -> Self {
        Self {
            neuron: neuron,
            inputs: RefCell::new(inputs.to_vec()),
        }
    }

    pub fn to_string(&self) -> String {
        let args: Vec<String> = self.inputs.borrow().iter().map(|v| match v {
            ConnectionValue::Value(value) => match value {
                NeuronValue::Char(s) => s.clone(),
                NeuronValue::Int32(i) => i.to_string(),
                NeuronValue::Int64(i) => i.to_string(),
                NeuronValue::Float(f) => f.to_string(),
                NeuronValue::Double(d) => d.to_string(),
                NeuronValue::Str(s) => s.clone(),
                NeuronValue::Type(t) => format!("{:?}", t),
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

    pub fn cost(&self) -> usize {
        let mut c = 0;
        let inputs = self.inputs.borrow();

        for v in inputs.iter() {

            c += 1;

            if let ConnectionValue::Connection(inner) = v {
                c += inner.cost();
            }
        }
        c
    }

    pub fn depth(&self, d: usize) -> usize {
        let inputs = self.inputs.borrow();

        inputs.iter().fold(d, |acc, v| {
            if let ConnectionValue::Connection(inner) = v {
                std::cmp::max(acc, inner.depth(d + 1))
            } else {
                acc
            }
        })
    }

    pub fn output(&self) -> Option<NeuronValue> {
        let mut args = Vec::with_capacity(self.inputs.borrow().len());
        let inputs = self.inputs.borrow();

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

        let mut inner_inputs = self.inputs.borrow_mut();
        let mut index: usize = 0;

        for i in 0..inner_inputs.len() {
            let input: &mut ConnectionValue = &mut inner_inputs[i];

            match input {
                ConnectionValue::Connection(conn_arc) => {
                    let mut conn = Arc::clone(conn_arc);
                    let size = conn.input_types().len();

                    if size > 0 {
                        let sub_inputs = &inputs[index..index + size];
                        let conn_mut = Arc::make_mut(&mut conn);

                        conn_mut.apply_inputs(sub_inputs);
                        *input = ConnectionValue::Connection(conn);
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
                                    NeuronValue::Type(t) => {
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

    pub fn input_types(&self) -> Vec<Type> {
        let inputs = self.inputs.borrow();
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
                        NeuronValue::Type(t) => {
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
