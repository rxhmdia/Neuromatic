using Neuromatic.Core;
using Neuromatic.Initializers;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;

namespace Neuromatic.TensorFlow
{
    /// <summary>
    /// A tensorflow implementation of the model backend
    /// </summary>
    public class TensorFlowModelBackend : ModelBackend
    {
        private TFSession _session;
        private Dictionary<string, TensorFlowGraphNode> _nodes;
        private List<ExecutableModelNode> _trainableWeights;
        
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of <see cref="TensorFlowModelBackend"/>
        /// </summary>
        public TensorFlowModelBackend()
        {
            _session = new TFSession();
            _nodes = new Dictionary<string, TensorFlowGraphNode>();
            _trainableWeights = new List<ExecutableModelNode>();
        }

        /// <summary>
        /// Gets the session used by the model backend
        /// </summary>
        public TFSession Session => _session;

        /// <summary>
        /// Gets the trainable weights
        /// </summary>
        public override IEnumerable<ExecutableModelNode> TrainableWeights => _trainableWeights;

        /// <summary>
        /// Adds a bias term to the given model node
        /// </summary>
        /// <param name="node">Node to add the bias term to</param>
        /// <param name="bias">Bias term to add</param>
        /// <returns>Returns the new model node with the bias term added</returns>
        public override ExecutableModelNode BiasAdd(ExecutableModelNode node, ExecutableModelNode bias)
        {
            string nodeName = Guid.NewGuid().ToString();

            return CreateNode(
                _session.Graph.BiasAdd(
                    ((TensorFlowGraphNode)node).Value,
                    ((TensorFlowGraphNode)bias).Value, operName: nodeName
                ), nodeName
            );
        }

        /// <summary>
        /// Creates a new constant value
        /// </summary>
        /// <param name="value">Value for the constant</param>
        /// <param name="name">Name of the constant</param>
        /// <returns>Returns the node for the constant value</returns>
        public override ExecutableModelNode Constant(object value, long[] shape, string name)
        {
            var constantDefinition = _session.Graph.Constant(value, new TFShape(shape), dtype: TFDataType.Float, operName: name);
            return CreateNode(constantDefinition, name);
        }

        /// <summary>
        /// Create a node that performs matrix multiplication
        /// </summary>
        /// <param name="left">Left node</param>
        /// <param name="right">Right node</param>
        /// <returns>Returns the matrix multiplication node</returns>
        public override ExecutableModelNode Dot(ExecutableModelNode left, ExecutableModelNode right, string name)
        {
            return CreateNode(
                _session.Graph.MatMul(
                    ((TensorFlowGraphNode)left).Value,
                    ((TensorFlowGraphNode)right).Value,
                    operName: name
                ),
                name
            );
        }

        /// <summary>
        /// Locates a node in the graph that is managed by the backend
        /// </summary>
        /// <param name="name">Name of the model node</param>
        /// <returns>Returns the found model node. Returns null when the node could not be found.</returns>
        public override ExecutableModelNode Node(string name)
        {
            if (_nodes.TryGetValue(name, out TensorFlowGraphNode node))
            {
                return node;
            }

            return null;
        }

        /// <summary>
        /// Creates a new input the model
        /// </summary>
        /// <param name="layer">Input definition</param>
        public override ExecutableModelNode Placeholder(string name, long[] shape)
        {
            if (_nodes.ContainsKey(name))
            {
                return _nodes[name];
            }

            return CreateNode(
                _session.Graph.Placeholder(TFDataType.Float, new TFShape(shape), name),
                name
            );
        }

        /// <summary>
        /// Initializes a tensor with a normal distribution
        /// </summary>
        /// <param name="shape">Shape of the tensor</param>
        /// <param name="mean">Mean value for the distribution</param>
        /// <param name="standardDeviation">Standard deviation</param>
        /// <param name="seed">Random seed to use</param>
        /// <returns></returns>
        public override ExecutableModelNode RandomNormal(long[] shape, float mean = 0, float standardDeviation = 0.05F, int? seed = null)
        {
            var nodeName = Guid.NewGuid().ToString();
            var operation = _session.Graph.RandomNormal(new TFShape(shape), mean, standardDeviation, seed, nodeName);

            return CreateNode(operation, nodeName);
        }

        /// <summary>
        /// Creates a new set of weights for the model
        /// </summary>
        /// <param name="initializer">Initializer to use</param>
        /// <param name="name">Name of the node</param>
        /// <returns>Returns the model node for the weights</returns>
        public override ExecutableModelNode Weights(long[] shape, InitializationFunction initializer, string name)
        {
            var operation = _session.Graph.Variable(
                ((TensorFlowGraphNode)initializer.Compile(shape, this)).Value,
                out TFOutput variableValue,
                operName: name);

            var weightNode = CreateNode(variableValue, name);

            _trainableWeights.Add(weightNode);

            return weightNode;
        }

        /// <summary>
        /// Disposes any managed resources that were used by this instance of the <see cref="TensorFlowModelBackend"/>
        /// </summary>
        /// <param name="disposing"></param>
        protected override void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                _session.Dispose();
            }

            _disposed = true;
        }

        /// <summary>
        /// Creates a functions that adds two inputs
        /// </summary>
        /// <param name="a">First input</param>
        /// <param name="b">Second input</param>
        /// <param name="name">Name of the operation</param>
        /// <returns>Returns the node for the add operation</returns>
        public override ExecutableModelNode Add(ExecutableModelNode a, ExecutableModelNode b, string name)
        {
            var operation = _session.Graph.Add(
                ((TensorFlowGraphNode)a).Value, 
                ((TensorFlowGraphNode)b).Value, 
                name);

            return CreateNode(operation, name);
        }

        /// <summary>
        /// Creates a function bound to the backend
        /// </summary>
        /// <param name="defaultInputs">The default values for all placeholders in the function</param>
        /// <param name="outputs">The list of outputs to fetch as part of the function</param>
        /// <returns>Returns an executable function</returns>
        public override BackendFunction Function(IEnumerable<ExecutableModelNode> inputs, IEnumerable<ExecutableModelNode> outputs, IEnumerable<ExecutableModelNode> updates)
        {
            return new TensorFlowBackendFunction(_session, inputs, outputs, updates);
        }

        /// <summary>
        /// Defines a functions that determines the gradients of the loss in relation to the variables.
        /// </summary>
        /// <param name="loss">Loss function</param>
        /// <param name="variables">Variables for the loss function</param>
        /// <returns>Returns the gradients for the loss in relation to the variables</returns>
        public override IEnumerable<ExecutableModelNode> Gradients(
            ExecutableModelNode loss, 
            IEnumerable<ExecutableModelNode> variables)
        {
            var outputs = _session.Graph.AddGradients(
                new[] { ((TensorFlowGraphNode)loss).Value },
                variables.Select(x => ((TensorFlowGraphNode)x).Value).ToArray());

            return outputs.Select(x => CreateNode(x));
        }

        /// <summary>
        /// Multiplies two values
        /// </summary>
        /// <param name="left">Left variable</param>
        /// <param name="right">Right variable</param>
        /// <returns>Returns the output of the multiplication operation</returns>
        public override ExecutableModelNode Multiply(ExecutableModelNode left, ExecutableModelNode right)
        {
            return CreateNode(_session.Graph.MatMul(Ref(left),Ref(right)));
        }

        /// <summary>
        /// Defines a variable with an integer value
        /// </summary>
        /// <param name="value">Initial value for the variable</param>
        /// <param name="name">Name of the variable</param>
        public override ExecutableModelNode Variable(int value, string name)
        {
            return CreateNode(_session.Graph.Variable(_session.Graph.Const(new TFTensor(value)), operName: name), name);
        }

        /// <summary>
        /// Adds a value to the value of an existing variable and assigns the new value to the existing variable.
        /// </summary>
        /// <param name="variable">Variable to update</param>
        /// <param name="value">Value to add</param>
        /// <returns>Returns the operator</returns>
        public override ExecutableModelNode UpdateAdd(ExecutableModelNode variable, ExecutableModelNode value)
        {
            return CreateNode(_session.Graph.AssignAdd(Ref(variable), Ref(value)));
        }

        /// <summary>
        /// Updates a variable with a new value
        /// </summary>
        /// <param name="w">Variable to update</param>
        /// <param name="newValue">The new value to assign to the variable</param>
        /// <returns>Returns the assignment operator</returns>
        public override ExecutableModelNode Update(ExecutableModelNode original, ExecutableModelNode newValue)
        {
            return CreateNode(_session.Graph.Assign(Ref(original), Ref(newValue)));
        }

        /// <summary>
        /// Creates a sigmoid function
        /// </summary>
        /// <param name="input">Input for the function</param>
        /// <returns>Output of the function</returns>
        public override ExecutableModelNode Sigmoid(ExecutableModelNode node)
        {
            return CreateNode(_session.Graph.Sigmoid(((TensorFlowGraphNode)node).Value));
        }

        /// <summary>
        /// Retrieves the original node reference from a executable model node
        /// </summary>
        /// <param name="node">Node to retrieve the graph reference from</param>
        /// <returns>Returns the raw reference</returns>
        private TFOutput Ref(ExecutableModelNode node)
        {
            return ((TensorFlowGraphNode)node).Value;
        }

        /// <summary>
        /// Creates a node around a raw TensorFlow symbol
        /// </summary>
        /// <param name="value">Raw symbol</param>
        /// <param name="name">Name that refers to the symbol</param>
        /// <returns>Returns the wrapped node</returns>
        private TensorFlowGraphNode CreateNode(TFOutput value, string name = null)
        {
            var node = new TensorFlowGraphNode(value);

            if (name != null)
            {
                // Store a named node in the nodes cache. This cache makes it easier to access specific nodes in the graph.
                // You need this to connect layers together and create proper metrics, losses and optimizer functions.
                _nodes[name] = node;
            }

            return node;
        }

        /// <summary>
        /// Clips the input by a minimum and maximum value
        /// </summary>
        /// <param name="input">Input to clip</param>
        /// <param name="minValue">Min value to clip to</param>
        /// <param name="maxValue">Max value to clip to</param>
        /// <returns>Returns the clipped value</returns>
        public override ExecutableModelNode ClipByValue(ExecutableModelNode input, float minValue, float maxValue)
        {
            return CreateNode(
                _session.Graph.ClipByValue(
                    Ref(input),
                    _session.Graph.Constant(minValue, new TFShape(1), TFDataType.Float),
                    _session.Graph.Constant(maxValue, new TFShape(1), TFDataType.Float)
                )
            );
        }

        /// <summary>
        /// Subtract two values
        /// </summary>
        /// <param name="left">Left node</param>
        /// <param name="right">Right node</param>
        /// <returns>Returns the subtraction operator</returns>
        public override ExecutableModelNode Subtract(ExecutableModelNode left, ExecutableModelNode right)
        {
            return CreateNode(
                _session.Graph.Sub(((TensorFlowGraphNode)left).Value,
                ((TensorFlowGraphNode)right).Value));
        }

        /// <summary>
        /// Performs a natural log on the input
        /// </summary>
        /// <param name="input"></param>
        /// <param name="subtract"></param>
        /// <returns></returns>
        public override ExecutableModelNode Log(ExecutableModelNode input)
        {
            return CreateNode(_session.Graph.Log(((TensorFlowGraphNode)input).Value));
        }

        /// <summary>
        /// Performs a division operation
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public override ExecutableModelNode Divide(ExecutableModelNode numerator, ExecutableModelNode divisor)
        {
            return CreateNode(
                _session.Graph.Div(
                    ((TensorFlowGraphNode)numerator).Value, 
                    ((TensorFlowGraphNode)divisor).Value
                )
            );
        }

        /// <summary>
        /// Calculates the sigmoid cross-entropy with logits
        /// </summary>
        /// <param name="target">Target tensor</param>
        /// <param name="result">Result tensor</param>
        /// <returns>Returns the outcome of the operation</returns>
        public override ExecutableModelNode SigmoidCrossEntropyWithLogits(ExecutableModelNode target, ExecutableModelNode result)
        {
            return CreateNode(
                _session.Graph.SigmoidCrossEntropyWithLogits(
                    ((TensorFlowGraphNode)target).Value, 
                    ((TensorFlowGraphNode)result).Value
                )
            );
        }

        public override ExecutableModelNode Float(ExecutableModelNode input)
        {
            return CreateNode(
                _session.Graph.Cast(((TensorFlowGraphNode)input).Value, TFDataType.Float));
        }
    }
}
