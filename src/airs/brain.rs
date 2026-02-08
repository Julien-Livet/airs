use rayon::prelude::*;
use itertools::{Itertools, MultiProduct};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::sync::Arc;

use crate::airs::connection;

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
                let input_types = connection.input_types();
                let mut args: Vec<Vec<ConnectionValue> > = Vec::new();
                
                for input_type in input_types {
                    let mut possibilities = Vec::new();
                    
                    possibilities.push(ConnectionValue::Value(NeuronValue::ValueType(input_type.clone())));

                    if let Some(existing_conns) = connection_mapping.get(&input_type.clone()) {
                        for existing in existing_conns {
                            possibilities.push(ConnectionValue::Connection(Arc::clone(existing)));
                        }
                    }

                    args.push(possibilities.clone());
                }
                
                for p in args.iter().multi_cartesian_product() {
                    let inputs: Vec<ConnectionValue> =
                        p.iter().cloned().cloned().collect();

                    let new_conn = Arc::new(connection.deep_clone());
                    new_conn.apply_inputs(&inputs);

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
        
        let mut conns = connections.clone().into_iter().collect::<Vec<_> >();
        conns.sort_by_key(|x| x.cost());
        
        let connection_args: Vec<(Arc<Connection>, Vec<Vec<ConnectionValue> >)> =
            conns
                .iter()
                .filter_map(|conn| {
                    let mut args = Vec::new();

                    for input_type in conn.input_types() {
                        let values = parameters.get(&input_type)?;
                        args.push(
                            values
                                .iter()
                                .cloned()
                                .map(ConnectionValue::Connection)
                                .collect()
                        );
                    }

                    Some((conn.clone(), args))
                })
                .collect();

        targets
            .par_iter()
            .map(|target| {
                let mut heap = BinaryHeap::new();

                'conn_loop: for (conn, args) in &connection_args {
                    for params in args.iter().multi_cartesian_product() {
                        let c = Arc::new(conn.deep_clone());
                        let inputs: Vec<ConnectionValue> = params.iter().cloned().cloned().collect();

                        c.apply_inputs(&inputs);

                        let cost = c
                            .output()
                            .map(|v| v.heuristic(target))
                            .unwrap_or(f64::INFINITY);

                        heap.push(Pair {
                            cost,
                            connection_cost: c.cost(),
                            connection: c,
                        });

                        if cost < eps {
                            break 'conn_loop;
                        }
                    }
                }

                heap.pop()
                    .expect("No solution found")
                    .connection
            })
            .collect()
    }
}
