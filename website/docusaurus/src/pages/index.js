import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <header className={clsx('hero', styles.heroBanner)}>
            <div className="container">
                <div className={styles.heroEquation}>
                    <span className={styles.yatSymbol}>ⵟ</span>
                    <span className={styles.equationText}>(𝐰, 𝐱) = </span>
                    <span className={styles.fraction}>
                        <span className={styles.numerator}>⟨𝐰, 𝐱⟩²</span>
                        <span className={styles.denominator}>‖𝐰 − 𝐱‖² + ε</span>
                    </span>
                </div>
                <Heading as="h1" className={styles.heroTitle}>
                    Neural Matter Networks
                </Heading>
                <p className={styles.heroSubtitle}>
                    Activation-Free Neural Computation with the <span className={styles.yatSymbol}>ⵟ</span>-Product
                </p>
                <p className={styles.heroDescription}>
                    The ⵟ-product unifies alignment and proximity in a single geometric operator,
                    enabling neural networks without activation functions while maintaining
                    universal approximation capabilities.
                </p>
                <div className={styles.buttons}>
                    <Link
                        className="button button--primary button--lg"
                        to="/docs/intro">
                        Get Started →
                    </Link>
                    <Link
                        className="button button--secondary button--lg"
                        href="https://github.com/mlnomadpy/nmn">
                        GitHub ⭐
                    </Link>
                    <Link
                        className="button button--secondary button--lg"
                        href="/paper/">
                        Interactive Paper 📄
                    </Link>
                </div>
            </div>
        </header>
    );
}

function Feature({ title, description, icon }) {
    return (
        <div className={clsx('col col--4', styles.feature)}>
            <div className={styles.featureCard}>
                <div className={styles.featureIcon}>{icon}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </div>
    );
}

function HomepageFeatures() {
    const features = [
        {
            title: 'Activation-Free',
            icon: '🎯',
            description: 'No ReLU, sigmoid, or tanh needed. The ⵟ-product provides intrinsic non-linearity through geometry.',
        },
        {
            title: 'Self-Regularizing',
            icon: '📉',
            description: 'Responses remain bounded and gradients decay naturally — no explicit normalization required.',
        },
        {
            title: 'Universal Approximation',
            icon: '∞',
            description: 'NMNs are dense in C(𝒳) under uniform norm on compact domains. Theoretical guarantees included.',
        },
        {
            title: 'Mercer Kernel',
            icon: '🎓',
            description: 'Symmetric and positive semidefinite, connecting to established kernel theory and RKHS.',
        },
        {
            title: 'Drop-in Replacement',
            icon: '🔄',
            description: 'NMN layers work as direct replacements for Linear + Activation in existing architectures.',
        },
        {
            title: 'Flax NNX Native',
            icon: '⚡',
            description: 'Built with Flax NNX for modern JAX-based deep learning with full pytree support.',
        },
    ];

    return (
        <section className={styles.features}>
            <div className="container">
                <div className="row">
                    {features.map((props, idx) => (
                        <Feature key={idx} {...props} />
                    ))}
                </div>
            </div>
        </section>
    );
}

function CodeExample() {
    return (
        <section className={styles.codeSection}>
            <div className="container">
                <h2 className={styles.sectionTitle}>Quick Example</h2>
                <div className={styles.codeBlock}>
                    <pre>
                        <code className="language-python">{`from flax import nnx
from nmn.nnx.nmn import YatNMN

# Create a YatNMN layer (replaces nn.Dense + activation)
layer = YatNMN(
    in_features=768,
    out_features=256,
    constant_alpha=True,  # Use sqrt(2) scaling
    rngs=nnx.Rngs(0)
)

# Forward pass - intrinsic non-linearity, no activation needed
y = layer(x)  # y = (x·W)² / (||x-W||² + ε)`}
                        </code>
                    </pre>
                </div>
            </div>
        </section>
    );
}

function Stats() {
    return (
        <section className={styles.stats}>
            <div className="container">
                <div className={styles.statsGrid}>
                    <div className={styles.statItem}>
                        <span className={styles.statValue}>15-25%</span>
                        <span className={styles.statLabel}>Memory Reduction</span>
                    </div>
                    <div className={styles.statItem}>
                        <span className={styles.statValue}>11.2%</span>
                        <span className={styles.statLabel}>Loss Improvement (BF16)</span>
                    </div>
                    <div className={styles.statItem}>
                        <span className={styles.statValue}>0</span>
                        <span className={styles.statLabel}>Activation Functions</span>
                    </div>
                </div>
            </div>
        </section>
    );
}

export default function Home() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout
            title={`${siteConfig.title}`}
            description="Activation-Free Neural Computation with the ⵟ-Product">
            <HomepageHeader />
            <main>
                <Stats />
                <HomepageFeatures />
                <CodeExample />
            </main>
        </Layout>
    );
}
