using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neutronal.Tensorflow
{
    /// <summary>
    /// A tensorflow implementation of the model backend
    /// </summary>
    public class TensorFlowModelBackend : ModelBackend
    {
        private TFSession _session;
        private Dictionary<string, TensorFlowGraphNode> _nodes;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of <see cref="TensorFlowModelBackend"/>
        /// </summary>
        public TensorFlowModelBackend()
        {
            _session = new TFSession();
            _nodes = new Dictionary<string, TensorFlowGraphNode>();
        }

        /// <summary>
        /// Gets the output for the currently compiled model
        /// </summary>
        public override ExecutableModel Output => throw new NotImplementedException();

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
            if(_nodes.TryGetValue(name, out TensorFlowGraphNode node))
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
            if(_nodes.ContainsKey(name))
            {
                return _nodes[name];
            }

            return CreateNode(
                _session.Graph.Placeholder(TFDataType.Float, new TFShape(shape), name), 
                name
            );
        }

        /// <summary>
        /// Creates a trainable variable
        /// </summary>
        /// <param name="name">Name of the variable</param>
        /// <returns>Returns the variable node</returns>
        public override ExecutableModelNode Variable(string name)
        {
            throw new NotImplementedException();
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
