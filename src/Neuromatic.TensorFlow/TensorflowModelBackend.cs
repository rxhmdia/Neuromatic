using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;
using Neutronal.Tensorflow;

namespace Neuromatic.TensorFlow
{
    /// <summary>
    /// A tensorflow implementation of the model backend
    /// </summary>
    public class TensorFlowModelBackend : ModelBackend
    {
        private TFSession _session;
        private Dictionary<string, TensorFlowGraphNode> _nodes;
        private TensorflowInitializers _initializers;
        private TensorFlowActivations _activations;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of <see cref="TensorFlowModelBackend"/>
        /// </summary>
        public TensorFlowModelBackend()
        {
            _session = new TFSession();
            _nodes = new Dictionary<string, TensorFlowGraphNode>();
            _activations = new TensorFlowActivations(this);
            _initializers = new TensorflowInitializers(this);
        }

        /// <summary>
        /// Gets the output for the currently compiled model
        /// </summary>
        public override ExecutableModel Output => throw new NotImplementedException();

        /// <summary>
        /// Gets the initializers supported by the model backend
        /// </summary>
        public override Initializers Initializers => _initializers;

        /// <summary>
        /// Gets the activation functions supported by the model backend
        /// </summary>
        public override Activations Activations => _activations;

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
                ((TensorFlowGraphNode)initializer.Create(shape)).Value,
                operName: name);

            return CreateNode(operation, name);
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

        private TensorFlowGraphNode CreateNode(TFOutput value, string name)
        {
            var node = new TensorFlowGraphNode(value);

            // Store the node in the nodes cache. This cache makes it easier to access specific nodes in the graph.
            // You need this to connect layers together and create proper metrics, losses and optimizer functions.
            _nodes[name] = node;

            return node;
        }
    }
}
