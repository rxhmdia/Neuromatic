using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Layers
{
    /// <summary>
    /// Defines a single layer in a neural network
    /// </summary>
    public abstract class Layer
    {
        /// <summary>
        /// Initializes a new instance of <see cref="Layer"/>
        /// </summary>
        /// <param name="name">Name of the layer</param>
        protected Layer(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Gets the name of the layer
        /// </summary>
        public string Name
        {
            get;
            internal set;
        }

        /// <summary>
        /// Gets the shape of the layer
        /// </summary>
        public abstract long[] Shape { get;}

        /// <summary>
        /// Compiles the layer
        /// </summary>
        /// <param name="backend">Backend to use for compilation</param>
        internal abstract void Compile(ModelBackend backend);
    }
}
