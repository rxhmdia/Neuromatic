using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Losses
{
    /// <summary>
    /// The mean-squared-error function calculates the distance between the target and predicted values.
    /// This loss function is typically used in regression scenarios where you predict continuous outputs
    /// based on one or more independent inputs.
    /// </summary>
    public class MeanSquaredError : LossFunction
    {
        /// <summary>
        /// Compiles the loss function
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="predictions">The output of the graph producing predictions</param>
        /// <param name="targets">The output of the graph containing the targets</param>
        /// <returns>Returns the compiled loss function</returns>
        public override TFOutput Compile(TFGraph graph, TFOutput predictions, TFOutput targets)
        {
            return graph.Mean(graph.Square(graph.Sub(predictions, targets)), graph.Const(-1));
        }
    }
}
