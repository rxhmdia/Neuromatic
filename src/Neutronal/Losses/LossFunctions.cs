using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Losses
{
    /// <summary>
    /// Defines a function that measures how well a model is doing towards a specific objective
    /// </summary>
    public abstract class LossFunction
    {
        /// <summary>
        /// Compiles the function using the provided model backend
        /// </summary>
        /// <param name="output">Output layer</param>
        /// <param name="backend">Model backend to use for compiling the loss function</param>
        public abstract void Compile(Layer output, ModelBackend backend);
    }
}
