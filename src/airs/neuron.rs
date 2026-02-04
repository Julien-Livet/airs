use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Char,
    Double,
    Float,
    Int64,
    String,
    Type
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Char(String),
    //Double(f64),
    //Float(f32),
    Int64(i64),
    Str(String),
    Type(Type),
}

impl Value {
    pub fn value_type(&self) -> Type {
        match self {
            Value::Char(_) => Type::Char,
            Value::Int64(_) => Type::Int64,
            Value::Str(_) => Type::String,
            Value::Type(t) => t.clone(),
        }
    }
}

pub type NeuronFn = dyn Fn(&[Value]) -> Option<Value> + Send + Sync;

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

    pub fn apply(&self, args: &[Value]) -> Option<Value> {
        (self.function)(args)
    }
}
