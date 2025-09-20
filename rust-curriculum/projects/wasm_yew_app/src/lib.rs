use wasm_bindgen::prelude::*;
use yew::prelude::*;

#[function_component(App)]
pub fn app() -> Html {
    let count = use_state(|| 0);
    let onclick = {
        let count = count.clone();
        Callback::from(move |_| count.set(*count + 1))
    };
    html! {
        <div>
            <h1>{"Yew Counter"}</h1>
            <button {onclick}>{"+1"}</button>
            <p>{ format!("Count: {}", *count) }</p>
        </div>
    }
}

#[wasm_bindgen(start)]
pub fn run() {
    yew::Renderer::<App>::new().render();
}
