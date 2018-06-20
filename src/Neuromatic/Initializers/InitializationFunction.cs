using System;
using TensorFlow;

namespace Neuromatic.Initializers
{
    /// <summary>
    /// Defines a function used to initialize variables in the model.
    /// For example, this is used in initializing weights and bias terms within layers.
    /// </summary>
    public abstract class InitializationFunction
    {
        /// <summary>
        /// Compiles the initialization function
        /// </summary>
        /// <param name="graph">Tensorflow graph to use for creating the initialization function</param>
        /// <param name="shape">The shape of the variable to initialize</param>
        /// <returns>Returns the compiled initialization function</returns>
        public abstract TFOutput Compile(TFGraph graph, TFShape shape);
    }
}
