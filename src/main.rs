mod airs;

#[cfg(not(test))]
fn main() {
}

#[cfg(test)]
mod tests
{
    use std::sync::Arc;

    //use super::airs::Brain as Brain;
    use super::airs::Connection as Connection;
    use super::airs::Neuron as Neuron;
    use super::airs::Type as Type;
    use super::airs::Value as Value;

    #[test]
    fn test_valid_connections() {
        let mut digit_neurons: Vec<Arc<Neuron> > = vec![];

        for i in 0..10 {
            let name = format!("{}", i);

            let neuron = Arc::new(Neuron::new(
                name,
                Arc::new(move |_inputs: &[Value]| {
                    Some(Value::Int64(i))
                }),
                vec![],
                Type::Int64,
            ));

            digit_neurons.push(neuron);
        }

        let add_neuron = Arc::new(Neuron::new(
            "add",
            Arc::new(|inputs: &[Value]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (Value::Int64(a), Value::Int64(b)) => {
                        Some(Value::Int64(a + b))
                    }
                    _ => None,
                }
            }),
            vec![Type::Int64, Type::Int64],
            Type::Int64,
        ));

        let sub_neuron = Arc::new(Neuron::new(
            "sub",
            Arc::new(|inputs: &[Value]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (Value::Int64(a), Value::Int64(b)) => {
                        Some(Value::Int64(a - b))
                    }
                    _ => None,
                }
            }),
            vec![Type::Int64, Type::Int64],
            Type::Int64,
        ));

        let mul_neuron = Arc::new(Neuron::new(
            "mul",
            Arc::new(|inputs: &[Value]| {
                if inputs.len() != 2 {
                    return None;
                }

                match (&inputs[0], &inputs[1]) {
                    (Value::Int64(a), Value::Int64(b)) => {
                        Some(Value::Int64(a * b))
                    }
                    _ => None,
                }
            }),
            vec![Type::Int64, Type::Int64],
            Type::Int64,
        ));

        let conn0 = Connection::new(
            digit_neurons[0].clone(),
            vec![],
        );

        assert_eq!(conn0.to_string(), "0");
        assert_eq!(conn0.output(), Some(Value::Int64(0)));
        assert_eq!(conn0.depth(0), 0);
        assert_eq!(conn0.cost(), 0);

        let mut digit_connections: Vec<Arc<Connection> > = vec![];

        for neuron in digit_neurons {
            digit_connections.push(Arc::new(Connection::new(neuron.clone(), vec![])))
        }

        let conn1 = Arc::new(Connection::new(add_neuron.clone(), [digit_connections[2].clone(), digit_connections[3].clone()].into_iter().collect()));

        assert_eq!(conn1.to_string(), "add(2, 3)");
        assert_eq!(conn1.output(), Some(Value::Int64(5)));
        assert_eq!(conn1.depth(0), 1);
        assert_eq!(conn1.cost(), 2);

        let /*mut*/ conn2 = Arc::new(Connection::new(mul_neuron.clone(), [conn1.clone(), digit_connections[4].clone()].into_iter().collect()));

        assert_eq!(conn2.to_string(), "mul(add(2, 3), 4)");
        assert_eq!(conn2.output(), Some(Value::Int64(20)));
        assert_eq!(conn2.depth(0), 2);
        assert_eq!(conn2.cost(), 4);
/*
        conn2.apply_inputs([digit_connections[3].clone(), digit_connections[5].clone(), digit_connections[4].clone()].into_iter().collect());
        assert_eq!(conn2.output(), Some(Value::Int64(32)));
*/
        let int_neuron = Arc::new(Neuron::new(
            "int",
            Arc::new(|_| Some(Value::Type(Type::Int64))),
            vec![],
            Type::Type,
        ));

        let int_connection = Arc::new(Connection::new(int_neuron.clone(), vec![]));

        let conn3 = Arc::new(Connection::new(sub_neuron.clone(), [int_connection.clone(), int_connection.clone()].into_iter().collect()));

        assert_eq!(conn3.to_string(), "sub(int, int)");
        assert_eq!(conn3.depth(0), 1);
        assert_eq!(conn3.cost(), 2);
    }

    #[test]
    fn test_str() {
        let mut neurons: Vec<Arc<Neuron> > = vec![];

        for i in 0..10 {
            let name = format!("{}", i);

            let neuron = Arc::new(Neuron::new(
                name,
                Arc::new(move |_inputs: &[Value]| {
                    Some(Value::Int64(i))
                }),
                vec![],
                Type::Int64,
            ));

            neurons.push(neuron);
        }

        //let brain: Brain = Brain::new(neurons);
        
        //let connections = brain.learn(["11"]);

        //assert_ne!(connections.len(), 0);

        //println!("{}", connections[0].to_string());
    }
}
