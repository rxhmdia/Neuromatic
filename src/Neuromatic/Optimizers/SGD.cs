using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Neuromatic.Core;

namespace Neuromatic.Optimizers
{
    /// <summary>
    /// Stochastic gradient descent optimizer
    /// </summary>
    public class SGD : Optimizer
    {
        /// <summary>
        /// Initializes a new instance of <see cref="SGD"/>
        /// </summary>
        /// <param name="learningRate">Learning rate for the optimizer</param>
        public SGD(float learningRate)
        {
            LearningRate = learningRate;
        }
        
        /// <summary>
        /// Gets the learning rate for the optimizer
        /// </summary>
        public float LearningRate { get; }

        /// <summary>
        /// Compiles the optimizer into a set of operations to be executed in each minibatch
        /// </summary>
        /// <param name="loss">Loss function to optimize</param>
        /// <param name="weights">The trainable model weights</param>
        /// <param name="backend">The backend used for compilation</param>
        /// <returns>Returns the compiled set of operations for the optimizer</returns>
        public override IEnumerable<ExecutableModelNode> Compile(ExecutableModelNode loss, IEnumerable<ExecutableModelNode> weights, ModelBackend backend)
        {
            var operations = new List<ExecutableModelNode>();

            var iterations = backend.Variable(0, "SGD_Iterations");
            var learningRate = backend.Constant(LearningRate, new long[] { 1 }, "SGD_LearningRate");
            var gradients = backend.Gradients(loss, weights);

            operations.Add(backend.UpdateAdd(iterations, backend.Constant(1, new long[] { 1 }, null)));

            foreach (var (w, g) in Enumerable.Zip(weights, gradients, (w, g) => (w, g)))
            {
                var velocity = backend.Multiply(learningRate, g);
                var newWeight = backend.Add(w, velocity);

                operations.Add(backend.Update(w, newWeight));
            }

            return operations;
        }

    }
}
