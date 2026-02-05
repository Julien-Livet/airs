use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Char,
    Double,
    Float,
    Int32,
    Int64,
    String,
    Type
}

#[derive(Clone, Debug, PartialEq)]
pub enum NeuronValue {
    Char(String),
    Double(f64),
    Float(f32),
    Int32(i32),
    Int64(i64),
    Str(String),
    Type(Type),
}

impl NeuronValue {
    pub fn value_type(&self) -> Type {
        match self {
            NeuronValue::Char(_) => Type::Char,
            NeuronValue::Double(_) => Type::Double,
            NeuronValue::Float(_) => Type::Float,
            NeuronValue::Int32(_) => Type::Int32,
            NeuronValue::Int64(_) => Type::Int64,
            NeuronValue::Str(_) => Type::String,
            NeuronValue::Type(t) => t.clone(),
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
