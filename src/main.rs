mod airs;

#[cfg(not(test))]
fn main() {
}

#[cfg(test)]
mod tests
{
    use std::collections::HashSet;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::Arc;

    use super::airs::Brain as Brain;
    use super::airs::Connection as Connection;
    use super::airs::ConnectionValue as ConnectionValue;
    use super::airs::Neuron as Neuron;
    use super::airs::ValueType as ValueType;
    use super::airs::NeuronValue as NeuronValue;

    #[test]
    fn test_valid_connections() {
        let mut digit_neurons: Vec<Arc<Neuron> > = vec![];
        
        for i in 0..10 {
            let name = format!("{}", i);

            let neuron = Arc::new(Neuron::new(
                name,
                Arc::new(move |_inputs: &[NeuronValue]| {
                    Some(NeuronValue::Int64(i))
                }),
                vec![],
                ValueType::Int64,
            ));

            digit_neurons.push(neuron);
        }
        
        let add_neuron = Arc::new(Neuron::new(
            "add",
            Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a + b))
                    }
                    _ => None,
                }
            }),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));
        
        let sub_neuron = Arc::new(Neuron::new(
            "sub",
            Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a - b))
                    }
                    _ => None,
                }
            }),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));
        
        let mul_neuron = Arc::new(Neuron::new(
            "mul",
            Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a * b))
                    }
                    _ => None,
                }
            }),
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
            Arc::new(|_| Some(NeuronValue::ValueType(ValueType::Int64))),
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
            Arc::new(|_| Some(NeuronValue::ValueType(ValueType::Int64))),
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
                Arc::new(move |_inputs: &[NeuronValue]| {
                    Some(NeuronValue::Int64(i))
                }),
                vec![],
                ValueType::Int64,
            ));

            neurons.push(neuron);
        }

        let add_neuron = Arc::new(Neuron::new(
            "add",
            Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a + b))
                    }
                    _ => None,
                }
            }),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));

        let mul_neuron = Arc::new(Neuron::new(
            "mul",
            Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (NeuronValue::Int64(a), NeuronValue::Int64(b)) => {
                        Some(NeuronValue::Int64(a * b))
                    }
                    _ => None,
                }
            }),
            vec![ValueType::Int64, ValueType::Int64],
            ValueType::Int64,
        ));

        let int_to_str_neuron = Arc::new(Neuron::new(
            "int_to_str",
            Arc::new(|inputs: &[NeuronValue]| {
                if inputs.len() != 1 {
                    return None;
                }

                match &inputs[0] {
                    NeuronValue::Int64(a) => {
                        Some(NeuronValue::String(format!("{}", a)))
                    }
                    _ => None,
                }
            }),
            vec![ValueType::Int64],
            ValueType::String,
        ));

        neurons.push(add_neuron);
        neurons.push(mul_neuron);
        neurons.push(int_to_str_neuron);

        let target = NeuronValue::String("11".into());
        let brain: Brain = Brain::new(neurons);
        let connections = brain.learn(&[target.clone()].to_vec(), 2);

        assert_ne!(connections.len(), 0);

        println!("{}", connections[0].to_string());

        assert!(connections[0].output().unwrap().heuristic(&target) == 0.0);
    }
}
