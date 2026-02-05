use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::sync::Arc;

use super::connection::Connection;
use super::connection::ConnectionValue;
use super::neuron::Neuron;
use super::neuron::NeuronValue;
use super::neuron::ValueType;
use super::utility::*;

pub struct Brain {
    neurons: Vec<Arc<Neuron> >,
}

#[derive(Clone)]
pub struct Pair {
    pub value: NeuronValue,
    pub cost: f64,
    pub connection_cost: usize,
    pub connection: Arc<Connection>,
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.connection_cost.cmp(&self.connection_cost))
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.connection_cost == other.connection_cost
    }
}

impl Eq for Pair {}

impl Brain {
    pub fn new(neurons: Vec<Arc<Neuron> >) -> Self {
        Self
        {
            neurons,
        }
    }

    pub fn learn(
        &self,
        targets: &[NeuronValue],
        max_level: usize,
        eps: f64,
    ) -> Vec<Arc<Connection> > {
        let mut connections: HashSet<Arc<Connection> > = Default::default();
        let mut parameters: HashMap<ValueType, Vec<Arc<Connection> > > = Default::default();

        for neuron in &self.neurons {
            let input_types = neuron.input_types();
            let output_type = neuron.output_type();

            if input_types.is_empty() {
                let conn = Arc::new(Connection::new(Arc::clone(neuron), &Vec::new()));
                
                parameters
                    .entry(output_type.clone())
                    .or_insert_with(Vec::new)
                    .push(conn);
            } else {
                let v: Vec<ConnectionValue> = input_types
                    .iter()
                    .map(|ty| ConnectionValue::Value(NeuronValue::ValueType(ty.clone())))
                    .collect();

                let conn = Arc::new(Connection::new(Arc::clone(neuron), &v));
                
                connections.insert(conn); 
            }
        }

        for neuron in &self.neurons {
            let input_types = neuron.input_types();
            let output_type = neuron.output_type();

            if input_types.is_empty() {
                let conn = Arc::new(Connection::new(Arc::clone(neuron), &Vec::new()));
                
                parameters
                    .entry(output_type.clone())
                    .or_insert_with(Vec::new)
                    .push(conn);
            }
        }

        let mut connection_mapping: HashMap<ValueType, HashSet<Arc<Connection> > > =
            HashMap::new();

        for _ in 0..max_level {
            let mut mapping = connection_mapping.clone();

            for connection in &connections {
                let neuron = connection.neuron();
                let output_type = neuron.output_type().clone();
                let input_types = neuron.input_types();
                let mut args: Vec<Vec<ConnectionValue> > = Vec::new();
                
                for input_type in input_types {
                    let mut possibilities = Vec::new();
                    
                    possibilities.push(ConnectionValue::Value(NeuronValue::ValueType(input_type.clone())));

                    if let Some(existing_conns) = connection_mapping.get(input_type) {
                        for existing in existing_conns {
                            possibilities.push(ConnectionValue::Connection(Arc::clone(existing)));
                        }
                    }

                    args.push(possibilities);
                }
                
                for p in cartesian_product(args) {
                    let new_conn = Arc::new(Connection::new(connection.neuron().clone(), &p));
                    
                    mapping
                        .entry(output_type.clone())
                        .or_insert_with(HashSet::new)
                        .insert(new_conn);
                }
            }

            connection_mapping = mapping;
            connections.clear();
            
            for set in connection_mapping.values() {
                for conn in set {
                    connections.insert(conn.clone());
                }
            }
        }
        for neuron in &self.neurons {
            if neuron.input_types().is_empty() {
                connections.insert(Arc::new(Connection::new(neuron.clone(), &vec![])));
            }
        }
        
        let connection_parameters: HashMap<Arc<Connection>, Vec<Vec<ConnectionValue> > > =
            connections
                .par_iter()
                .filter_map(|conn| {
                    let input_types = conn.input_types();
                    let mut args: Vec<Vec<ConnectionValue>> = Vec::new();

                    for input_type in input_types {
                        if let Some(values) = parameters.get(&input_type) {
                            args.push(values.clone().into_iter().map(ConnectionValue::Connection).collect::<Vec<_> >());
                        } else {
                            return None;
                        }
                    }

                    let product = cartesian_product(args);

                    Some((conn.clone(), product))
                })
                .collect();
            
        if connection_parameters.is_empty() {
            return vec![];
        }

        targets
            .par_iter()
            .map(|target| {
                let mut heap = BinaryHeap::new();

                'conn_loop: for (conn, params_list) in &connection_parameters {
                    for params in params_list {
                        let c = Arc::new(conn.deep_clone());

                        c.apply_inputs(params);

                        let value = c.output().unwrap();
                        let cost = value.heuristic(target);

                        heap.push(Pair {
                            value,
                            cost,
                            connection_cost: c.cost(),
                            connection: c,
                        });

                        if cost < eps {
                            break 'conn_loop;
                        }
                    }
                }

                heap
                    .pop()
                    .expect("No solution found")
                    .connection
            })
            .collect()
    }
}
