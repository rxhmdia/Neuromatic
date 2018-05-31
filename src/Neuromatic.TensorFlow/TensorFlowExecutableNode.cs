using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neutronal.Tensorflow
{
    /// <summary>
    /// Defines a reference to a Tensorflow symbol
    /// </summary>
    public class TensorFlowGraphNode : ExecutableModelNode
    {
        /// <summary>
        /// Initializes a new instance of <see cref="TensorFlowGraphNode"/>
        /// </summary>
        /// <param name="value"></param>
        public TensorFlowGraphNode(TFOutput value)
        {
            Value = value;
        }

        /// <summary>
        /// Gets the value of the node
        /// </summary>
        public TFOutput Value { get; }
    }
}
