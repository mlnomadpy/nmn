import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
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
                    Activation-free <span className={styles.yatSymbol}>ⵟ</span>YAT layers.
                    One library, six frameworks, numerically equivalent.
                </p>
                <p className={styles.heroDescription}>
                    The <span className={styles.yatSymbol}>ⵟ</span>-product fuses alignment and
                    proximity into a single geometric kernel, giving you non-linearity without
                    activation functions. Ship the same YatNMN math across PyTorch, Flax NNX,
                    Flax Linen, Keras, TensorFlow, and MLX.
                </p>
                <div className={styles.buttons}>
                    <Link
                        className="button button--primary button--lg"
                        to="/docs/intro">
                        Get Started →
                    </Link>
                    <Link
                        className="button button--secondary button--lg"
                        href="https://github.com/azettaai/nmn">
                        GitHub ⭐
                    </Link>
                    <Link
                        className="button button--secondary button--lg"
                        href="/paper/">
                        Interactive Paper 📄
                    </Link>
                </div>
                <div className={styles.install}>
                    <code>pip install nmn</code>
                </div>
            </div>
        </header>
    );
}

function Frameworks() {
    const frameworks = [
        { name: 'PyTorch', module: 'nmn.torch' },
        { name: 'Flax NNX', module: 'nmn.nnx' },
        { name: 'Flax Linen', module: 'nmn.linen' },
        { name: 'Keras', module: 'nmn.keras' },
        { name: 'TensorFlow', module: 'nmn.tf' },
        { name: 'MLX', module: 'nmn.mlx' },
    ];
    return (
        <section className={styles.frameworks}>
            <div className="container">
                <h2 className={styles.sectionTitle}>Six frameworks, one kernel</h2>
                <p className={styles.sectionSubtitle}>
                    Each backend is an independent, optional install. The YAT math stays
                    numerically equivalent across all of them.
                </p>
                <div className={styles.frameworkGrid}>
                    {frameworks.map((fw) => (
                        <div key={fw.name} className={styles.frameworkChip}>
                            <span className={styles.frameworkName}>{fw.name}</span>
                            <code className={styles.frameworkModule}>{fw.module}</code>
                        </div>
                    ))}
                </div>
            </div>
        </section>
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
            title: 'Six Frameworks',
            icon: '🧩',
            description: 'PyTorch, Flax NNX, Flax Linen, Keras, TensorFlow, and MLX — independent optional installs, numerically equivalent YAT layers everywhere.',
        },
        {
            title: 'YAT Geometric Kernel',
            icon: '📐',
            description: 'The ⵟ-product unifies alignment and proximity into one operator — a symmetric, positive-semidefinite kernel that removes the need for activation functions.',
        },
        {
            title: 'O(n) Linear Attention',
            icon: '⚡',
            description: 'MAY (Maclaurin) and RAY (radial) performer feature maps give linear-time YAT attention. Bias-aware and sign-indefinite — MAY beats bias-free SLAY at bias b > 0.',
        },
        {
            title: 'Lazy Mode',
            icon: '🧊',
            description: 'lazy=True / freeze_kernel=True freezes only the kernel while bias, alpha, and epsilon stay trainable — cheap adaptation from pretrained weights.',
        },
        {
            title: 'nmn CLI',
            icon: '🛠️',
            description: 'nmn doctor, nmn guide <framework>, nmn features, and nmn frameworks help you discover backends and verify your install from the terminal.',
        },
        {
            title: 'Drop-in Replacement',
            icon: '🔄',
            description: 'YatNMN layers slot in wherever you use Linear + Activation, with universal-approximation guarantees backed by kernel theory.',
        },
    ];

    return (
        <section className={styles.features}>
            <div className="container">
                <h2 className={styles.sectionTitle}>Why NMN</h2>
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
                <p className={styles.sectionSubtitle}>
                    A YatNMN layer replaces Linear + activation. Same math, whichever
                    framework you reach for.
                </p>
                <div className={styles.codeBlock}>
                    <pre>
                        <code className="language-python">{`from flax import nnx
from nmn.nnx.nmn import YatNMN

# A YatNMN layer replaces nn.Dense + activation
layer = YatNMN(
    in_features=768,
    out_features=256,
    rngs=nnx.Rngs(0),
)

# Forward pass — intrinsic non-linearity, no activation needed
y = layer(x)  # y = ⟨x, W⟩² / (||x - W||² + ε)`}
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
                        <span className={styles.statValue}>6</span>
                        <span className={styles.statLabel}>Frameworks</span>
                    </div>
                    <div className={styles.statItem}>
                        <span className={styles.statValue}>O(n)</span>
                        <span className={styles.statLabel}>MAY / RAY Attention</span>
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
            description="Activation-free YAT layers. One library, six frameworks, numerically equivalent.">
            <HomepageHeader />
            <main>
                <Stats />
                <Frameworks />
                <HomepageFeatures />
                <CodeExample />
            </main>
        </Layout>
    );
}
