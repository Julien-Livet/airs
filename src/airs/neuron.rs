use std::fmt::{Debug, Display, Error, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::utility::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
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
    ValueType(ValueType),
}

impl Display for NeuronValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self)
    }
}

impl PartialEq for NeuronValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NeuronValue::Bool(a), NeuronValue::Bool(b)) => a == b,
            (NeuronValue::Char(a), NeuronValue::Char(b)) => a == b,
            (NeuronValue::Int64(a), NeuronValue::Int64(b)) => a == b,
            (NeuronValue::Int32(a), NeuronValue::Int32(b)) => a == b,
            (NeuronValue::Float(a), NeuronValue::Float(b)) => {
                a.to_bits() == b.to_bits()
            }
            (NeuronValue::Double(a), NeuronValue::Double(b)) => {
                a.to_bits() == b.to_bits()
            }
            (NeuronValue::String(a), NeuronValue::String(b)) => a == b,
            (NeuronValue::ValueType(a), NeuronValue::ValueType(b)) => a == b,
            (NeuronValue::Grid(a), NeuronValue::Grid(b)) => a == b,
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
            NeuronValue::ValueType(v) => v.hash(state),
            NeuronValue::String(v) => v.hash(state),
        }
    }
}

impl NeuronValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            NeuronValue::Bool(_) => ValueType::Bool,
            NeuronValue::Char(_) => ValueType::Char,
            NeuronValue::Double(_) =>ValueType::Double,
            NeuronValue::Float(_) => ValueType::Float,
            NeuronValue::Int32(_) => ValueType::Int32,
            NeuronValue::Int64(_) => ValueType::Int64,
            NeuronValue::String(_) => ValueType::String,
            NeuronValue::Grid(_) => ValueType::Grid,
            NeuronValue::ValueType(t) => t.clone(),
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
    input_types: Vec<ValueType>,
    output_type: ValueType,
}

impl Debug for Neuron {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Neuron")
         .field("name", &self.name)
         .field("input_types", &self.input_types)
         .field("output_type", &self.output_type)
         .finish()
    }
}

impl Neuron {
    pub fn new(
        name: impl Into<String>,
        function: Arc<NeuronFn>,
        input_types: Vec<ValueType>,
        output_type: ValueType,
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

    pub fn input_types(&self) -> &[ValueType] {
        &self.input_types
    }

    pub fn output_type(&self) -> &ValueType {
        &self.output_type
    }

    pub fn apply(&self, args: &[NeuronValue]) -> Option<NeuronValue> {
        (self.function)(args)
    }
}
