use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::utility::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Bool,
    Char,
    Double,
    Float,
    Int32,
    Int64,
    String,
    Grid,
    Type,
}

#[derive(Clone, Debug)]
pub enum NeuronValue {
    Bool(bool),
    Char(String),
    Double(f64),
    Float(f32),
    Int32(i32),
    Int64(i64),
    String(String),
    Grid(Vec<Vec<u8> >),
    Type(Type),
}

impl PartialEq for NeuronValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NeuronValue::Bool(a), NeuronValue::Bool(b)) => a == b,
            (NeuronValue::Int64(a), NeuronValue::Int64(b)) => a == b,
            (NeuronValue::Int32(a), NeuronValue::Int32(b)) => a == b,
            (NeuronValue::Float(a), NeuronValue::Float(b)) => {
                a.to_bits() == b.to_bits()
            }
            (NeuronValue::Double(a), NeuronValue::Double(b)) => {
                a.to_bits() == b.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for NeuronValue {}

impl Hash for NeuronValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            NeuronValue::Bool(v) => v.hash(state),
            NeuronValue::Char(v) => v.hash(state),
            NeuronValue::Int32(v) => v.hash(state),
            NeuronValue::Int64(v) => v.hash(state),
            NeuronValue::Float(v) => v.to_bits().hash(state),
            NeuronValue::Double(v) => v.to_bits().hash(state),
            NeuronValue::Grid(v) => v.hash(state),
            NeuronValue::Type(v) => v.hash(state),
            NeuronValue::String(v) => v.hash(state),
        }
    }
}

impl NeuronValue {
    pub fn value_type(&self) -> Type {
        match self {
            NeuronValue::Bool(_) => Type::Bool,
            NeuronValue::Char(_) => Type::Char,
            NeuronValue::Double(_) => Type::Double,
            NeuronValue::Float(_) => Type::Float,
            NeuronValue::Int32(_) => Type::Int32,
            NeuronValue::Int64(_) => Type::Int64,
            NeuronValue::String(_) => Type::String,
            NeuronValue::Grid(_) => Type::Grid,
            NeuronValue::Type(t) => t.clone(),
        }
    }

    pub fn heuristic(&self, target: &NeuronValue) -> f64 {
        match (self, target) {
            (NeuronValue::Bool(a), NeuronValue::Bool(b)) => (*a as i32 - *b as i32).abs() as f64,
            (NeuronValue::Char(a), NeuronValue::Char(b)) => levenshtein(a, b) as f64,
            (NeuronValue::Int64(a), NeuronValue::Int64(b)) => (*a - *b).abs() as f64,
            (NeuronValue::Int32(a), NeuronValue::Int32(b)) => (*a - *b).abs() as f64,
            (NeuronValue::Float(a), NeuronValue::Float(b)) => (*a - *b).abs() as f64,
            (NeuronValue::Double(a), NeuronValue::Double(b)) => (*a - *b).abs(),
            (NeuronValue::String(a), NeuronValue::String(b)) => levenshtein(a, b) as f64,
            (NeuronValue::Grid(a), NeuronValue::Grid(b)) => {
                if a.len() != b.len() || a[0].len() != b[0].len() {
                    return f64::INFINITY;
                }
                a.iter().zip(b).fold(0.0, |acc, (ra, rb)| {
                    acc + ra.iter().zip(rb).map(|(x,y)| (*x as f64 - *y as f64).abs()).sum::<f64>()
                })
            }
            _ => f64::INFINITY,
        }
    }
}

pub type NeuronFn = dyn Fn(&[NeuronValue]) -> Option<NeuronValue> + Send + Sync;

pub struct Neuron {
    name: String,
    function: Arc<NeuronFn>,
    input_types: Vec<Type>,
    output_type: Type,
}

impl Neuron {
    pub fn new(
        name: impl Into<String>,
        function: Arc<NeuronFn>,
        input_types: Vec<Type>,
        output_type: Type,
    ) -> Self {
        Self {
            name: name.into(),
            function,
            input_types,
            output_type,
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn input_types(&self) -> &[Type] {
        &self.input_types
    }

    pub fn output_type(&self) -> &Type {
        &self.output_type
    }

    pub fn apply(&self, args: &[NeuronValue]) -> Option<NeuronValue> {
        (self.function)(args)
    }
}
