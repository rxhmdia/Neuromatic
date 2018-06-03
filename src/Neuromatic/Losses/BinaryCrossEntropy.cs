using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;
using TensorFlow;

namespace Neuromatic.Losses
{
    /// <summary>
    /// Calculates the binary cross-entropy loss between the actual values and the provided target values
    /// </summary>
    public class BinaryCrossEntropy : LossFunction
    {
        /// <summary>
        /// Compiles the function using the provided model backend
        /// </summary>
        /// <param name="output">Output layer</param>
        /// <param name="backend">Model backend to use for compiling the loss function</param>
        public override ExecutableModelNode Compile(
            ExecutableModelNode output,
            ExecutableModelNode target,
            ModelBackend backend)
        {
            // Clip the value to make sure that the output stays stable.
            var result = backend.ClipByValue(output, Constants.Epsilon, 1 - Constants.Epsilon);

            // y = log(output / 1 - output)
            result = backend.Log(backend.Divide(result, backend.Subtract(
                backend.Constant(1, new long[] { 1 }, null), result)));

            return backend.SigmoidCrossEntropyWithLogits(target, result);
        }
    }
}
