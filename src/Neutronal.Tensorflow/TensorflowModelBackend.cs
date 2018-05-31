using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neutronal.Tensorflow
{
    /// <summary>
    /// A tensorflow implementation of the model backend
    /// </summary>
    public class TensorflowModelBackend : ModelBackend
    {
        private bool disposed;

        /// <summary>
        /// Gets the output for the currently compiled model
        /// </summary>
        public override ExecutableModel Output => throw new NotImplementedException();

        /// <summary>
        /// Creates a new Tensorflow graph for the given model
        /// </summary>
        /// <param name="model"></param>
        public override void CreateModel(Model model)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Creates a new input the model
        /// </summary>
        /// <param name="layer">Input definition</param>
        public override void Input(Input layer)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Disposes any managed resources that were used by this instance of the <see cref="TensorflowModelBackend"/>
        /// </summary>
        /// <param name="disposing"></param>
        protected override void Dispose(bool disposing)
        {
            if(disposed)
            {
                return;
            }

            if(disposing)
            {
                //TODO: Dispose managed resources
            }

            disposed = true;
        }

    }
}
