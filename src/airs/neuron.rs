use ndarray::Array2;
use std::fmt::{Debug, Display, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

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
    Grids,
    Type,
    Map,
    PairGrids,
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
    Grid(Array2<i8>),
    Grids(Vec<Array2<i8> >),
    ValueType(ValueType),
    Map(HashMap<i8, i8>),
    PairGrids(Vec<(Array2<i8>, Array2<i8>)>),
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
            (NeuronValue::Grids(a), NeuronValue::Grids(b)) => a == b,
            (NeuronValue::Map(a), NeuronValue::Map(b)) => a == b,
            (NeuronValue::PairGrids(a), NeuronValue::PairGrids(b)) => a == b,
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
            NeuronValue::Grids(v) => v.hash(state),
            NeuronValue::ValueType(v) => v.hash(state),
            NeuronValue::String(v) => v.hash(state),
            NeuronValue::Map(v) => v.iter().collect::<Vec<_> >().hash(state),
            NeuronValue::PairGrids(v) => v.hash(state),
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
            NeuronValue::Grids(_) => ValueType::Grids,
            NeuronValue::ValueType(t) => t.clone(),
            NeuronValue::Map(_) => ValueType::Map,
            NeuronValue::PairGrids(_) => ValueType::PairGrids,
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
            (NeuronValue::Grid(val), NeuronValue::Grid(target)) => {
                if val.shape() != target.shape() {
                    return 100.0 + (val.sum() - target.sum()).abs() as f64;
                }

                val.iter()
                    .zip(target.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<i8>() as f64
            }
            (NeuronValue::Grids(val), NeuronValue::Grids(target)) => {
                if val.len() != target.len() {
                    return 100.0 + (val.len() as f64 - target.len() as f64).abs();
                }

                let x: Vec<f64> = val.iter()
                    .zip(target.iter())
                    .map(|(a, b)| NeuronValue::Grid(a.clone()).heuristic(&NeuronValue::Grid(b.clone())))
                    .collect();

                x.iter().map(|d| d * d).sum::<f64>().sqrt()
            }
            _ => f64::INFINITY,
        }
    }
}

pub type NeuronFn = dyn Fn(&[NeuronValue]) -> Option<NeuronValue> + Send + Sync;

pub struct Neuron {
    name: String,
    pub function: RwLock<Arc<NeuronFn> >,
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
        function: RwLock<Arc<NeuronFn> >,
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
        let func = self.function.read().unwrap();

        (func)(args)
    }
}
