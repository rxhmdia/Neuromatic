using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;

namespace Neuromatic.Layers
{
    public class Dense : Layer
    {
        public Dense(string name, long[] shape, Layer input) : base(name)
        {
            Shape = shape;
            Input = input;
        }

        public override long[] Shape { get; }

        public Layer Input { get;}

        /// <summary>
        /// Compiles the dense layer
        /// </summary>
        /// <param name="backend">Backend to use for performing compilation</param>
        /// <returns>Returns the compiled layer output node</returns>
        internal override ExecutableModelNode Compile(ModelBackend backend)
        {
            return backend.Dot(Input.Compile(backend), backend.Variable($"{Name}_Weights"));
        }
    }
}
