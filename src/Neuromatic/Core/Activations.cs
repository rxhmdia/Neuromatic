using Neuromatic.Activations;
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
        private ModelBackend _backend;

        /// <summary>
        /// Initializes a new instance of <see cref="Activations"/>
        /// </summary>
        /// <param name="backend"></param>
        public Activations(ModelBackend backend)
        {
            _backend = backend;
        }

        /// <summary>
        /// Creates a sigmoid activation function
        /// </summary>
        /// <param name="input">Input node for the function</param>
        /// <returns>Output node of the function</returns>
        public abstract ActivationFunction Sigmoid();
    }
}
