import React, {useEffect} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function PaperRedirect() {
    const target = useBaseUrl('/paper/index.html');

    useEffect(() => {
        window.location.replace(target);
    }, [target]);

    return <main>Opening the interactive NMN paper…</main>;
}
