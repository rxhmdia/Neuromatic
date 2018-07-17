using System;
using TensorFlow;

namespace Neuromatic.Losses
{
    /// <summary>
    /// Defines a function that calculates the loss for an output of a model
    /// </summary>
    public abstract class LossFunction
    {
        /// <summary>
        /// Compiles the loss function
        /// </summary>
        /// <param name="context">Compilation context to use</param>
        /// <param name="predictions">The output of the graph producing predictions</param>
        /// <param name="targets">The output of the graph containing the targets</param>
        /// <returns>Returns the compiled loss function</returns>
        public abstract TFOutput Compile(ModelCompilationContext context, TFOutput predictions, TFOutput targets);
    }
}
