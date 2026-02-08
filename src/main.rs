mod airs;
mod arc;
mod primitives;

use ndarray::Array2;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use airs::Neuron as Neuron;
use airs::NeuronFn as NeuronFn;
use airs::ValueType as ValueType;

use crate::airs::NeuronValue;

fn update_region_neurons(region_neurons: &mut HashMap<i8, Arc<Neuron> >, pairs: &Vec<Array2<i8> >) {
    let mut region_map: HashMap<i8, Vec<Vec<Vec<(isize, isize)> > > > = HashMap::new();

    for input_ in pairs {
        let s = airs::region_set(input_, false);

        let mut regions: HashMap<i8, Vec<Vec<(isize, isize)> > > = (0..10)
            .map(|i| (i, Vec::new()))
            .collect();

        for r in s {
            let value = input_[(r[0].0 as usize, r[0].1 as usize)];
            regions.get_mut(&value).unwrap().push(r);
        }

        for (&k, v) in &regions {
            let l = region_map.entry(k).or_insert_with(Vec::new);
            l.push(v.clone());
        }
    }

    for (&k, v) in &region_map {
        let function: Arc<NeuronFn> = Arc::new({
            let v = v.clone();

            move |_| Some(NeuronValue::RegionsList(v.clone()))
        });

        let empty_count = v.iter().filter(|l| l.is_empty()).count();

        match region_neurons.get_mut(&k) {
            Some(neuron) => {
                if empty_count != v.len() {
                    let mut func = neuron.function.write().unwrap();
                    *func = function;
                } else {
                    region_neurons.remove(&k);
                }
            }
            None => {
                if empty_count != v.len() {
                    region_neurons.insert(
                        k,
                        Arc::new(Neuron::new(
                            format!("region{}", k),
                            RwLock::new(function),
                            vec![],
                            ValueType::RegionsList
                        )),
                    );
                }
            }
        }
    }
}

#[cfg(not(test))]
fn main() {
}

#[cfg(test)]
mod tests
{
    use std::collections::{HashMap, HashSet};
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::{Arc, RwLock};

    use super::airs::Brain as Brain;
    use super::airs::Connection as Connection;
    use super::airs::ConnectionValue as ConnectionValue;
    use super::airs::Neuron as Neuron;
    use super::airs::ValueType as ValueType;
    use super::airs::NeuronValue as NeuronValue;

    use super::arc::load_task;
    use super::arc::input_output_pairs;
    use super::arc::input_option_output_pairs;
    use super::primitives::*;

    #[test]
    fn test_valid_connections() {
        let mut digit_neurons: Vec<Arc<Neuron> > = vec![];
        
        for i in 0..10 {
            let name = format!("{}", i);

            let neuron = Arc::new(Neuron::new(
                name,
                RwLock::new(Arc::new(move |_inputs: &[NeuronValue]| {
                    Some(NeuronValue::Int64(i))
                })),
                vec![],
                ValueType::Int64,
            ));

            digit_neurons.push(neuron);
        }
        
        let add_neuron = Arc::new(Neuron::new(
            "add",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a + b))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));
        
        let sub_neuron = Arc::new(Neuron::new(
            "sub",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a - b))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));
        
        let mul_neuron = Arc::new(Neuron::new(
            "mul",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a * b))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));

        let conn0 = Connection::new(
            digit_neurons[0].clone(),
            &vec![],
        );

        assert_eq!(conn0.to_string(), "0");
        assert_eq!(conn0.output(), Some(NeuronValue::Int64(0)));
        assert_eq!(conn0.depth(0), 0);
        assert_eq!(conn0.cost(), 0);

        let conn1 = Arc::new(Connection::new(add_neuron.clone(),
        &[ConnectionValue::Value(NeuronValue::Int64(2)), ConnectionValue::Value(NeuronValue::Int64(3))].to_vec()));

        assert_eq!(conn1.to_string(), "add(2, 3)");
        assert_eq!(conn1.output(), Some(NeuronValue::Int64(5)));
        assert_eq!(conn1.depth(0), 0);
        assert_eq!(conn1.cost(), 2);

        let conn2 = Arc::new(Connection::new(mul_neuron.clone(),
        &[ConnectionValue::Connection(conn1.clone()), ConnectionValue::Value(NeuronValue::Int64(4))].to_vec()));

        assert_eq!(conn2.to_string(), "mul(add(2, 3), 4)");
        assert_eq!(conn2.output(), Some(NeuronValue::Int64(20)));
        assert_eq!(conn2.depth(0), 1);
        assert_eq!(conn2.cost(), 4);

        conn2.apply_inputs(&[ConnectionValue::Value(NeuronValue::Int64(3)), ConnectionValue::Value(NeuronValue::Int64(5)), ConnectionValue::Value(NeuronValue::Int64(4))].to_vec());
        assert_eq!(conn2.output(), Some(NeuronValue::Int64(32)));

        let int_neuron = Arc::new(Neuron::new(
            "int",
            RwLock::new(Arc::new(|_| {
                Some(NeuronValue::ValueType(ValueType::Int64))
            })),
            vec![],
            ValueType::Type,
        ));

        let int_connection = Arc::new(Connection::new(int_neuron.clone(), &vec![]));

        let conn3 = Arc::new(Connection::new(sub_neuron.clone(),
        &[ConnectionValue::Connection(int_connection.clone()), ConnectionValue::Connection(int_connection.clone())].to_vec()));

        assert_eq!(conn3.to_string(), "sub(int, int)");
        assert_eq!(conn3.depth(0), 1);
        assert_eq!(conn3.cost(), 2);
    }

    #[test]
    fn test_connection_eq()
    {
        let int_neuron = Arc::new(Neuron::new(
            "int",
            RwLock::new(Arc::new(|_: &[NeuronValue]| {
                Some(NeuronValue::ValueType(ValueType::Int64))
            })),
            vec![],
            ValueType::Type,
        ));

        let conn1 = Arc::new(Connection::new(int_neuron.clone(), &vec![]));
        let conn2 = Arc::new(Connection::new(int_neuron.clone(), &vec![]));

        assert_eq!(conn1, conn2);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        conn1.hash(&mut h1);
        conn1.hash(&mut h2);
        
        assert_eq!(h1.finish(), h2.finish());

        let mut connections: HashSet<Arc<Connection> > = Default::default();

        connections.insert(conn1.clone());
        connections.insert(conn1.clone());
        connections.insert(conn2.clone());

        assert_eq!(connections.len(), 1);
    }

    #[test]
    fn test_str() {
        let mut neurons: Vec<Arc<Neuron> > = vec![];

        for i in 0..10 {
            let name = format!("{}", i);

            let neuron = Arc::new(Neuron::new(
                name,
                RwLock::new(Arc::new(move |_inputs: &[NeuronValue]| {
                    Some(NeuronValue::Int64(i))
                })),
                vec![],
                ValueType::Int64,
            ));

            neurons.push(neuron);
        }

        let add_neuron = Arc::new(Neuron::new(
            "add",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a + b))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));

        let mul_neuron = Arc::new(Neuron::new(
            "mul",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a * b))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));

        let int_to_str_neuron = Arc::new(Neuron::new(
            "int_to_str",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::Int64(a) => {
                        Some(NeuronValue::String(format!("{}", a)))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Int64],
            ValueType::String,
        ));

        neurons.push(add_neuron.clone());
        neurons.push(mul_neuron.clone());
        neurons.push(int_to_str_neuron.clone());

        let target = NeuronValue::String("11".into());

        let conn2 = Arc::new(Connection::new(neurons[2].clone(), &vec![]));
        let conn9 = Arc::new(Connection::new(neurons[9].clone(), &vec![]));
        let conn = Arc::new(Connection::new(int_to_str_neuron, &[ConnectionValue::Connection(Arc::new(Connection::new(add_neuron.clone(), &[ConnectionValue::Connection(conn2), ConnectionValue::Connection(conn9)])))]));

        assert!(conn.output().unwrap().heuristic(&target) == 0.0);

        let brain: Brain = Brain::new(neurons);
        let connections = brain.learn(&[target.clone()].to_vec(), 2, 1e-6);

        assert_ne!(connections.len(), 0);

        println!("{}", connections[0].to_string());

        assert!(connections[0].output().unwrap().heuristic(&target) == 0.0);
    }

    #[tokio::test]
    async fn test_task3c9b0459() -> Result<(), Box<dyn std::error::Error> > {
        let task = load_task("training", "3c9b0459").await?;
        let train_pairs = input_output_pairs(&task.train);
        let test_pairs = input_option_output_pairs(&task.test);
        
        let fliplr_neuron = Arc::new(Neuron::new(
            "fliplr",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::Grids(a) => {
                        Some(NeuronValue::Grids(fliplr(a)))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Grids],
            ValueType::Grids,
        ));

        let flipud_neuron = Arc::new(Neuron::new(
            "flipud",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::Grids(a) => {
                        Some(NeuronValue::Grids(flipud(a)))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Grids],
            ValueType::Grids,
        ));

        let mut input = train_pairs.inputs;

        let input_neuron = Arc::new(Neuron::new(
            "input",
            RwLock::new(Arc::new(move |_inputs| {
                Some(NeuronValue::Grids(input.clone()))
            })),
            vec![],
            ValueType::Grids,
        ));

        let mut neurons: Vec<Arc<Neuron> > = vec![];

        neurons.push(fliplr_neuron);
        neurons.push(flipud_neuron);
        neurons.push(input_neuron.clone());

        let target = NeuronValue::Grids(train_pairs.outputs);

        let brain: Brain = Brain::new(neurons);
        let connections = brain.learn(&[target.clone()].to_vec(), 2, 1e-6);

        assert_ne!(connections.len(), 0);

        println!("{}", connections[0].to_string());

        assert!(connections[0].output().unwrap().heuristic(&target) == 0.0);
        
        input = test_pairs.inputs;
        
        {
            let mut func = input_neuron.function.write().unwrap();
            *func = Arc::new(move |_inputs: &[NeuronValue]| {
                Some(NeuronValue::Grids(input.clone()))
            });
        }
        
        assert!(connections[0].output().unwrap().heuristic(&NeuronValue::Grids(test_pairs.outputs)) == 0.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_task0d3d703e() -> Result<(), Box<dyn std::error::Error> > {
        let task = load_task("training", "0d3d703e").await?;
        let train_pairs = input_output_pairs(&task.train);
        let test_pairs = input_option_output_pairs(&task.test);

        let infer_color_mapping_neuron = Arc::new(Neuron::new(
            "infer_color_mapping",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::PairGrids(a) => {
                        Some(NeuronValue::Map(infer_color_mapping(a)))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::PairGrids],
            ValueType::Map,
        ));

        let map_neuron = Arc::new(Neuron::new(
            "map",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Grids(a), NeuronValue::Map(b)) => {
                        Some(NeuronValue::Grids(map(a, b)))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Grids, ValueType::Map],
            ValueType::Grids,
        ));

        let pair_grids = task.train;

        let train_pairs_neuron = Arc::new(Neuron::new(
            "train_pairs",
            RwLock::new(Arc::new(move |_inputs| {
                Some(NeuronValue::PairGrids(pair_grids.clone()))
            })),
            vec![],
            ValueType::PairGrids,
        ));

        let output_grids: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i8>, ndarray::Dim<[usize; 2]>, i8>> = train_pairs.outputs.clone();

        let pairs_neuron = Arc::new(Neuron::new(
            "pairs",
            RwLock::new(Arc::new(move |inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::Grids(a) => {
                        if a.len() != output_grids.len() {
                            return None;
                        }
                        
                        let mut v = vec![];
                        
                        for i in 0..a.len() {
                            v.push((a[i].clone(), output_grids[i].clone()));
                        }

                        Some(NeuronValue::PairGrids(v))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Grids],
            ValueType::PairGrids,
        ));

        let mut input = train_pairs.inputs;

        let input_neuron = Arc::new(Neuron::new(
            "input",
            RwLock::new(Arc::new(move |_inputs| {
                Some(NeuronValue::Grids(input.clone()))
            })),
            vec![],
            ValueType::Grids,
        ));

        let mut neurons: Vec<Arc<Neuron> > = vec![];

        neurons.push(infer_color_mapping_neuron.clone());
        neurons.push(map_neuron.clone());
        neurons.push(train_pairs_neuron.clone());
        //neurons.push(pairs_neuron.clone());
        neurons.push(input_neuron.clone());

        let target = NeuronValue::Grids(train_pairs.outputs);

        let brain: Brain = Brain::new(neurons);
        let connections = brain.learn(&[target.clone()].to_vec(), 2, 1e-6);

        assert_ne!(connections.len(), 0);

        println!("{}", connections[0].to_string());

        assert!(connections[0].output().unwrap().heuristic(&target) == 0.0);
        
        input = test_pairs.inputs;
        
        {
            let mut func = input_neuron.function.write().unwrap();
            *func = Arc::new(move |_inputs: &[NeuronValue]| {
                Some(NeuronValue::Grids(input.clone()))
            });
        }
        
        assert!(connections[0].output().unwrap().heuristic(&NeuronValue::Grids(test_pairs.outputs)) == 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_task253bf280() -> Result<(), Box<dyn std::error::Error> > {
        let task = load_task("training", "253bf280").await?;
        let train_pairs = input_output_pairs(&task.train);
        let test_pairs = input_option_output_pairs(&task.test);

        let input = train_pairs.clone().inputs;

        let input_neuron = Arc::new(Neuron::new(
            "input",
            RwLock::new(Arc::new(move |_inputs| {
                Some(NeuronValue::Grids(input.clone()))
            })),
            vec![],
            ValueType::Grids,
        ));

        let segments_neuron = Arc::new(Neuron::new(
            "segments",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 5 {
                    return None;
                }

                match (&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]) {
                    (NeuronValue::Grids(a), NeuronValue::LocationPairs(b), NeuronValue::Int8(c), NeuronValue::Bool(d), NeuronValue::Bool(e)) => {
                        Some(NeuronValue::Grids(segments(a, b, c.clone(), d.clone(), e.clone())))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::Grids, ValueType::LocationPairs, ValueType::Int8, ValueType::Bool, ValueType::Bool],
            ValueType::Grids,
        ));

        let same_element_neuron = Arc::new(Neuron::new(
            "same_element",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::LocationPairs(a), NeuronValue::Bool(b)) => {
                        Some(NeuronValue::LocationPairs(same_element(a, b.clone())))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::LocationPairs, ValueType::Bool],
            ValueType::LocationPairs,
        ));

        let region_pairs_neuron = Arc::new(Neuron::new(
            "region_pairs",
            RwLock::new(Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::RegionsList(a) => {
                        Some(NeuronValue::LocationPairs(region_pairs(&a.clone())))
                    }
                    _ => None,
                }
            })),
            vec![ValueType::RegionsList],
            ValueType::LocationPairs,
        ));

        let false_neuron = Arc::new(Neuron::new(
            "false",
            RwLock::new(Arc::new(move |_inputs| {
                Some(NeuronValue::Bool(false))
            })),
            vec![],
            ValueType::Bool,
        ));

        let true_neuron = Arc::new(Neuron::new(
            "true",
            RwLock::new(Arc::new(move |_inputs| {
                Some(NeuronValue::Bool(true))
            })),
            vec![],
            ValueType::Bool,
        ));

        let mut digit_neurons: Vec<Arc<Neuron> > = vec![];
        
        for i in 0..10 {
            let name = format!("{}", i);

            let neuron = Arc::new(Neuron::new(
                name,
                RwLock::new(Arc::new(move |_inputs: &[NeuronValue]| {
                    Some(NeuronValue::Int8(i as i8))
                })),
                vec![],
                ValueType::Int8,
            ));

            digit_neurons.push(neuron);
        }

        let mut region_neurons : HashMap<i8, Arc<Neuron> > = HashMap::new();
        let train_inputs = train_pairs.clone().inputs;
        super::update_region_neurons(&mut region_neurons, &train_inputs.clone());

        let mut neurons: Vec<Arc<Neuron> > = vec![];

        neurons.push(segments_neuron.clone());
        neurons.push(same_element_neuron.clone());
        neurons.push(region_pairs_neuron.clone());
        neurons.push(false_neuron.clone());
        neurons.push(true_neuron.clone());
        neurons.push(input_neuron.clone());

        for (_k, v) in &region_neurons {
            neurons.push(v.clone());
        }

        {
            let mut digits: HashSet<i8> = HashSet::new();

            for m in &train_pairs.inputs {
                for &v in m.iter() {
                    digits.insert(v);
                }
            }

            for m in &train_pairs.outputs {
                for &v in m.iter() {
                    digits.insert(v);
                }
            }

            let mut existing_digits = Vec::with_capacity(10);
            for i in 0..10 {
                existing_digits.push(digits.contains(&(i as i8)));
            }

            for (i, exists) in existing_digits.iter().enumerate() {
                if *exists {
                    neurons.push(digit_neurons[i].clone());
                }
            }
        }

        let target = NeuronValue::Grids(train_pairs.outputs);

        let brain: Brain = Brain::new(neurons);
        let connections = brain.learn(&[target.clone()].to_vec(), 3, 1e-6);

        assert_ne!(connections.len(), 0);

        println!("{}", connections[0].to_string());

        assert!(connections[0].output().unwrap().heuristic(&target) == 0.0);
        
        let test_inputs = test_pairs.clone().inputs;
        super::update_region_neurons(&mut region_neurons, &test_inputs);

        {
            let mut func = input_neuron.function.write().unwrap();
            *func = Arc::new(move |_inputs: &[NeuronValue]| {
                Some(NeuronValue::Grids(test_inputs.clone()))
            });
        }
        
        assert!(connections[0].output().unwrap().heuristic(&NeuronValue::Grids(test_pairs.outputs)) == 0.0);

        Ok(())
    }
}
