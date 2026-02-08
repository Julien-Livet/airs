use rayon::prelude::*;
use itertools::{Itertools, MultiProduct};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::airs::connection;

use super::connection::Connection;
use super::connection::ConnectionValue;
use super::neuron::Neuron;
use super::neuron::NeuronValue;
use super::neuron::ValueType;
use super::utility::*;

fn load_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(std::sync::atomic::Ordering::Relaxed))
}

fn store_min(a: &AtomicU64, value: f64) {
    let mut old = a.load(std::sync::atomic::Ordering::Relaxed);

    loop {
        let old_f = f64::from_bits(old);
        if value >= old_f {
            break;
        }

        match a.compare_exchange_weak(
            old,
            value.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old = x,
        }
    }
}

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

        let global_best = Arc::new(AtomicU64::new(f64::INFINITY.to_bits()));

        targets
            .par_iter()
            .map(|target| {
                let local_best = Arc::new(AtomicU64::new(f64::INFINITY.to_bits()));

                connection_args
                    .par_iter()
                    .flat_map_iter( |(conn, args)| {
                        let conn = conn.clone();
                        let global_best = Arc::clone(&global_best);
                        let local_best = Arc::clone(&local_best);

                        args.iter()
                            .multi_cartesian_product()
                            .filter_map(move |params| {
                                if load_f64(&global_best) < eps {
                                    return None;
                                }

                                let inputs: Vec<ConnectionValue> = params.iter().cloned().cloned().collect();
                                let cost = conn
                                    .output_with_inputs(&inputs)
                                    .map(|v| v.heuristic(target))
                                    .unwrap_or(f64::INFINITY);

                                if cost >= load_f64(&local_best) {
                                    return None;
                                }

                                store_min(&local_best, cost);
                                store_min(&global_best, cost);

                                let new_conn = conn.clone();
                                new_conn.apply_inputs(&inputs);

                                Some(Pair {
                                    cost,
                                    connection_cost: conn.cost(),
                                    connection: new_conn,
                                })
                            })
                    })
                    .reduce_with(|a, b| if a.cost < b.cost { a } else { b })
                    .expect("No solution found")
                    .connection
            })
            .collect()
    }
}
