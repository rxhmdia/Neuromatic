using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Provides several activation functions
    /// </summary>
    public abstract class Activations
    {
        /// <summary>
        /// Creates a sigmoid activation function
        /// </summary>
        /// <param name="input">Input node for the function</param>
        /// <returns>Output node of the function</returns>
        public abstract ActivationFunction Sigmoid();
    }
}
