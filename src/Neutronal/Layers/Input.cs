using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Layers
{
    /// <summary>
    /// Defines an input for a model. 
    /// Inputs are used to feed data into a model.
    /// </summary>
    public class Input : Layer
    {
        /// <summary>
        /// Initializes a new instance of <see cref="Input"/>
        /// </summary>
        /// <param name="elements">Number of neurons in the input layer</param>
        /// <param name="name">Name of the input layer</param>
        public Input(long[] shape, string name = null) : base(name)
        {
            Shape = shape;
        }

        /// <summary>
        /// Gets the number of elements in the input layer
        /// </summary>
        public override long[] Shape { get; }

        /// <summary>
        /// Compiles the input layer
        /// </summary>
        /// <param name="backend">Backend to use for compilation</param>
        internal override void Compile(ModelBackend backend)
        {
            backend.Input(this);
        }
    }
}
